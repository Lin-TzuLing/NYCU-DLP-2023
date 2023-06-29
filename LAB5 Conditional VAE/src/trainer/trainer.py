import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from others.utils import plot_pred, plot_rec, finn_eval_seq, pred, mse_metric, kl_criterion
import matplotlib.pyplot as plt



class kl_annealing():
    def __init__(self, args):
        super().__init__()

        self.args = args
        # (bool) use cyclical mode
        self.kl_anneal_cyclical = self.args.kl_anneal_cyclical
        # (float) The decay ratio of kl annealing
        # (the proportion of a period that do linear kl_beta increasing)
        self.kl_anneal_ratio = self.args.kl_anneal_ratio
        # (int) The number of cycle for kl annealing during training (if use cyclical mode)
        self.kl_anneal_cycle = self.args.kl_anneal_cycle
        # initial kl beta (update later)
        self.kl_beta = self.args.beta

        # lower-bound and upper-bound of kl_beta
        self.start = self.args.beta
        self.stop = 1

        self.period = int(self.args.niter / self.kl_anneal_cycle)
        # kl_beta += step for each epoch in (period*kl_anneal_ratio)
        self.step = (self.stop - self.start) / (self.period * self.kl_anneal_ratio)
        self.mono_step = (self.stop - self.start) / (self.args.niter * self.kl_anneal_ratio)


    def update(self, epoch):
        if self.kl_anneal_cyclical:
            if (epoch % self.period) == 0:
                self.kl_beta = self.start
            else:
                if (epoch % self.period) <= (self.period * self.kl_anneal_ratio):
                    self.kl_beta += self.step
        else:
            if epoch <= (self.args.niter*self.kl_anneal_ratio):
                self.kl_beta += self.mono_step

    def get_beta(self):
        return min(self.kl_beta, 1.0)

    def plot(self):
        fig = plt.figure()
        ax1 = plt.subplot(211)

        self.kl_anneal_cyclical = False
        epochs = []
        kl_history = []
        for i in range(300):
            self.update(i)
            epochs.append(i)
            kl_history.append(self.get_beta())
        plt.plot(epochs, kl_history)

        ax2 = plt.subplot(212, sharex=ax1)
        self.kl_anneal_cyclical = True
        self.kl_beta = self.start
        epochs = []
        kl_history = []
        for i in range(300):
            self.update(i)
            epochs.append(i)
            kl_history.append(self.get_beta())
        plt.plot(epochs, kl_history)
        plt.show()
        return kl_history


mse_criterion = nn.MSELoss()
class Trainer():
    def __init__(self, args, frame_predictor, posterior, encoder, decoder, device):
        super(Trainer).__init__()

        self.args = args
        self.device = device

        # ---------------- optimizers ----------------
        if self.args.optimizer == 'adam':
            self.args.optimizer = optim.Adam
        elif self.args.optimizer == 'rmsprop':
            self.args.optimizer = optim.RMSprop
        elif self.args.optimizer == 'sgd':
            self.args.optimizer = optim.SGD
        else:
            raise ValueError('Unknown optimizer: %s' % self.args.optimizer)

        self.modules = {
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'encoder': encoder,
            'decoder': decoder,
        }

        params = list(self.modules['frame_predictor'].parameters()) + \
                      list(self.modules['posterior'].parameters()) + \
                      list(self.modules['encoder'].parameters()) + \
                      list(self.modules['decoder'].parameters())
        self.optimizer = self.args.optimizer(params, lr=self.args.lr, betas=(self.args.beta1, 0.999))
        self.kl_anneal = kl_annealing(self.args)


    def train(self, start_epoch, niter,
              train_data, train_loader, train_iterator,
              validate_data, validate_loader, validate_iterator):

        # kl_history = self.kl_anneal.plot()
        progress = tqdm(total=self.args.niter)
        best_val_psnr = 0
        kl_beta_history, tfr_history = [], []
        for epoch in range(start_epoch, start_epoch + niter):
            self.modules["frame_predictor"].train()
            self.modules["posterior"].train()
            self.modules["encoder"].train()
            self.modules["decoder"].train()

            epoch_loss = 0
            epoch_mse = 0
            epoch_kld = 0
            ### Update kl beta ###
            self.kl_anneal.update(epoch=epoch)

            # batch training
            for _ in range(self.args.epoch_size):
                try:
                    seq, cond = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    seq, cond = next(train_iterator)

                # swap batch (dim 0) and frame (dim 1), since h_seq is frame sequence
                seq = seq.transpose(0, 1)
                cond = cond.transpose(0, 1)

                loss, mse, kld = self.train_batch(seq, cond)
                epoch_loss += loss
                epoch_mse += mse
                epoch_kld += kld

            progress.update(1)
            with open('./{}/train_record.txt'.format(self.args.log_dir), 'a') as train_record:
                train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f'
                                    ' | tf ratio: %.5f | kl beta: %.5f\n' %
                (epoch, epoch_loss / self.args.epoch_size, epoch_mse / self.args.epoch_size, epoch_kld / self.args.epoch_size,
                 self.args.tfr, self.kl_anneal.get_beta())))

            if epoch >= self.args.tfr_start_decay_epoch:
                ### Update teacher forcing ratio ###
                # decay step size of teacher forcing ratio (0 ~ 1)
                new_tfr = self.args.tfr - min(self.args.tfr_decay_step, 1.0/self.args.niter)
                # lower bound of teacher forcing ratio for scheduling
                if new_tfr >= self.args.tfr_lower_bound:
                    self.args.tfr = new_tfr

            ### record kl beta, tfr ###
            kl_beta_history.append(self.kl_anneal.get_beta())
            tfr_history.append(self.args.tfr)


            self.modules["frame_predictor"].eval()
            self.modules["posterior"].eval()
            self.modules["encoder"].eval()
            self.modules["decoder"].eval()

            if epoch % 5 == 0:
                psnr_list = []
                for _ in range(len(validate_data) // self.args.batch_size):
                    try:
                        validate_seq, validate_cond = next(validate_iterator)
                    except StopIteration:
                        validate_iterator = iter(validate_loader)
                        validate_seq, validate_cond = next(validate_iterator)

                    # swap batch (dim 0) and frame (dim 1), since h_seq is frame sequence
                    validate_seq = validate_seq.transpose(0, 1)
                    validate_cond = validate_cond.transpose(0, 1)
                    # validation (predict n_future frames)
                    pred_seq = pred(validate_seq, validate_cond, self.modules, self.args, self.device)
                    # only evaluate predicted frames (without n_past)
                    _, _, psnr = finn_eval_seq(validate_seq[self.args.n_past:self.args.n_past+self.args.n_future],
                                               pred_seq[self.args.n_past:])
                    psnr_list.append(psnr)

                ave_psnr = np.mean(np.concatenate(psnr))

                with open('./{}/train_record.txt'.format(self.args.log_dir), 'a') as train_record:
                    train_record.write(
                        ('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

                if ave_psnr > best_val_psnr:
                    best_val_psnr = ave_psnr
                    # save the model
                    torch.save({
                        'encoder': self.modules["encoder"],
                        'decoder': self.modules["decoder"],
                        'frame_predictor': self.modules["frame_predictor"],
                        'posterior': self.modules["posterior"],
                        'args': self.args,
                        'last_epoch': epoch},
                        '%s/model.pth' % self.args.log_dir)

            if epoch % 20 == 0:
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)

                # swap batch (dim 0) and frame (dim 1), since h_seq is frame sequence
                validate_seq = validate_seq.transpose(0, 1)
                validate_cond = validate_cond.transpose(0, 1)
                plot_pred(validate_seq, validate_cond, self.modules, epoch, self.args, self.device)
                plot_rec(validate_seq, validate_cond, self.modules, epoch, self.args, self.device)

    def train_batch(self, x, cond):
        """train batch-loop"""

        self.modules['frame_predictor'].zero_grad()
        self.modules['posterior'].zero_grad()
        self.modules['encoder'].zero_grad()
        self.modules['decoder'].zero_grad()

        # initialize the hidden state.
        self.modules['frame_predictor'].hidden = self.modules['frame_predictor'].init_hidden()
        self.modules['posterior'].hidden = self.modules['posterior'].init_hidden()
        mse = 0
        kld = 0
        use_teacher_forcing = True if random.random() < self.args.tfr else False

        """data to device (sequence (x) and condition (cond))"""
        x = x.to(self.device)
        cond = cond.to(self.device)

        """encode all sequence first (avoid repeatedly encode in next loop)"""
        h_seq = [self.modules['encoder'](x[i]) for i in range(self.args.n_past + self.args.n_future)]

        """predict 1~12 frames"""
        # x_pred = None
        for frame_idx in range(1, self.args.n_past + self.args.n_future):
            """h = x_(t-1), h_target = x_t"""

            # h_target = input for prior lstm (posterior)
            h_target, _ = h_seq[frame_idx]
            # h, skip = input for lstm before decoder (frame_predictor)
            # skip last frame or condition on n_past frame (must use teacher forcing with skip)
            if self.args.last_frame_skip or (frame_idx < self.args.n_past):
                h, skip =  h_seq[frame_idx-1]
            else:
                if use_teacher_forcing:
                    h, _ = h_seq[frame_idx-1]
                else:
                    h, _  = self.modules['encoder'](x_pred)

            # construct latent code z of h_1
            z_target, mu, logvar = self.modules['posterior'](h_target)

            # input to lstm before decoder (frame_predictor)
            lstm_in = torch.concat([h, z_target, cond[frame_idx-1]], dim=1)
            lstm_out = self.modules['frame_predictor'](lstm_in)
            x_pred = self.modules['decoder']((lstm_out, skip))

            # reconstruction loss
            mse += mse_criterion(x_pred, x[frame_idx])
            # KL divergence
            kld += kl_criterion(mu, logvar, self.args)

        beta = self.kl_anneal.get_beta()
        loss = mse + kld * beta
        loss.backward()

        self.optimizer.step()

        return loss.detach().cpu().numpy() / (self.args.n_past + self.args.n_future), mse.detach().cpu().numpy() / \
               (self.args.n_past + self.args.n_future), kld.detach().cpu().numpy() / (self.args.n_future + self.args.n_past)

    def test(self, test_data, test_loader, test_iterator):
        psnr_list = []
        for _ in range(len(test_data) // self.args.batch_size):
            try:
                test_seq, test_cond = next(test_iterator)
            except StopIteration:
                test_iterator = iter(test_loader)
                test_seq, test_cond = next(test_iterator)

            # swap batch (dim 0) and frame (dim 1), since h_seq is frame sequence
            test_seq = test_seq.transpose(0, 1)
            test_cond = test_cond.transpose(0, 1)
            # validation (predict n_future frames)
            pred_seq = pred(test_seq, test_cond, self.modules, self.args, self.device)
            # only evaluate predicted frames (without n_past)
            _, _, psnr = finn_eval_seq(test_seq[self.args.n_past:self.args.n_past + self.args.n_future],
                                       pred_seq[self.args.n_past:])
            psnr_list.append(psnr)

        avg_psnr = np.mean(np.concatenate(psnr))
        print("test psnr = {:.5f}".format(avg_psnr))

        # sample = np.random.randint(self.args.batch_size)
        sample = 0
        plot_pred(test_seq, test_cond, self.modules, "test", self.args, self.device, batch_idx=sample)
        plot_rec(test_seq, test_cond, self.modules, "test", self.args, self.device, batch_idx=sample)
