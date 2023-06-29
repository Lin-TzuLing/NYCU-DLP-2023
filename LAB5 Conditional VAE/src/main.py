import argparse
import os
import random
import torch

from dataLoader.dataloader import load_train_data, load_test_data
from models.build_model import build_model
from trainer.trainer import Trainer

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=0, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')

    # self-defined
    parser.add_argument("--mode", type=str, choices=['train','test'], help="train or test")
    parser.add_argument("--cond_dim", type=int, default=7, help="dimensionality of condition vector")
    # parser.add_argument('--use_mean', default=True, action='store_true', help="use mean as latent z in validation")
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    # ------------ set device  -------------
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'

    # ------------ make assert  -------------
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    # ------------ load models  -------------
    if args.model_dir != '':
        mode = args.mode
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        if mode == "train":
            args.log_dir = '%s/continued' % args.log_dir
        elif mode == "test":
            args.log_dir = '%s/test' % args.log_dir
        start_epoch = saved_model['last_epoch']
        args.mode = mode
    else:
        saved_model = None
        if args.exp_name==None:
            name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
                % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)
        else:
            name = args.exp_name
        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    # ------------ set seed and makedirs  -------------
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  -------------
    frame_predictor, posterior, encoder, decoder = build_model(args, saved_model, device)

    # --------- load a dataset ------------------------------------
    if args.mode=="train":
        train_data, train_loader, train_iterator, \
            validate_data, validate_loader, validate_iterator = load_train_data(args)
    elif args.mode=="test":
        test_data, test_loader, test_iterator = load_test_data(args)

    # --------- set trainer ------------------------------------
    if args.mode == "train":
        trainer = Trainer(args, frame_predictor, posterior, encoder, decoder, device)
        trainer.train(start_epoch, niter,
                      train_data, train_loader, train_iterator,
                      validate_data, validate_loader, validate_iterator)
        print("done training")
    elif args.mode == "test":
        print("start testing")
        assert args.model_dir != ''
        trainer = Trainer(args, frame_predictor, posterior, encoder, decoder, device)
        trainer.test(test_data, test_loader, test_iterator)
        print("done testing")
    print()


        
