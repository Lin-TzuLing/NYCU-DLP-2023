import os
import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image, make_grid


def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def plot_rec(validate_seq, validate_cond, modules, epoch, args, device, batch_idx=0):
    """plot prediction result with latent code z sampled from prior"""
    pred_seq = pred_rec(validate_seq, validate_cond, modules, args, device)
    path = "{}/img/epoch_{}_rec".format(args.log_dir, epoch)
    print("epoch {}, saving reconstructed images ...".format(epoch))
    os.makedirs(path, exist_ok=True)

    # plot 12 frames (dim 0) in first batch (dim 1)
    # plot_seq = prediction, true_seq = plot in plot_pred
    images, pred_frames = [], []
    pred_seq = pred_seq[:, batch_idx, :, :, :]
    for frame_idx in range(len(pred_seq)):
        # HWC format for images list
        img_file = os.path.join(path, "{}.png".format(frame_idx))
        save_image(pred_seq[frame_idx], img_file)
        images.append(imageio.v2.imread(img_file))
        os.remove(img_file)
        # NCHW format tensor for pred_frames, true_frames list
        pred_frames.append(pred_seq[frame_idx])

    # continuous frame to one image
    pred_grid = make_grid(pred_frames, nrow=len(pred_seq))
    save_image(pred_grid, os.path.join(path, "pred_grid.png"))
    # gif
    imageio.mimsave(os.path.join(path, "animation.gif"), images)



def plot_pred(validate_seq, validate_cond, modules, epoch, args, device, batch_idx=0):
    """plot prediction result with latent code z sampled from N(0,I)"""
    pred_seq = pred(validate_seq, validate_cond, modules, args, device)
    path = "{}/img/epoch_{}_pred".format(args.log_dir, epoch)
    print("epoch {}, saving predicted images ...".format(epoch))
    os.makedirs(path, exist_ok=True)

    # plot 12 frames (dim 0) in first batch (dim 1)
    # plot_seq = prediction, true_seq = validation ground truth
    images, pred_frames, true_frames = [], [], []
    pred_seq, true_seq = pred_seq[:, batch_idx, :, :, :], validate_seq[:, batch_idx, :, :, :]
    for frame_idx in range(len(pred_seq)):
        # HWC format for images list
        img_file = os.path.join(path,"{}.png".format(frame_idx))
        save_image(pred_seq[frame_idx], img_file)
        images.append(imageio.v2.imread(img_file))
        os.remove(img_file)
        # NCHW format tensor for pred_frames, true_frames list
        pred_frames.append(pred_seq[frame_idx])
        true_frames.append(true_seq[frame_idx])

    # continuous frame to one image
    pred_grid = make_grid(pred_frames, nrow=len(pred_seq))
    true_grid = make_grid(true_frames, nrow=len(true_seq))
    save_image(pred_grid, os.path.join(path,"pred_grid.png"))
    save_image(true_grid, os.path.join(path, "true_grid.png"))
    # gif
    imageio.mimsave(os.path.join(path, "animation.gif"), images)



def pred(validate_seq, validate_cond, modules, args, device):
    """batch prediction for validation"""

    """data to device (sequence (x) and condition (cond))"""
    x = validate_seq.to(device)
    cond = validate_cond.to(device)

    """generate valid prediction"""
    with torch.no_grad():
        # initialize the hidden state.
        modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
        modules['posterior'].hidden = modules['posterior'].init_hidden()

        """predict 1~12 frames"""
        # initial input image x_in = x[0]
        gen_seq = []
        gen_seq.append(x[0])
        x_in = x[0]
        for frame_idx in range(1, args.n_past + args.n_future):
            # encode input image h at timestep (t-1) => (frame_idx-1)
            if args.last_frame_skip or (frame_idx < args.n_past):
                h, skip =  modules['encoder'](x_in)
            else:
                h, _  = modules['encoder'](x_in)

            # get latent code z at timestep (t) => (frame_idx)
            if (frame_idx < args.n_past):
                h_target, _ = modules['encoder'](x[frame_idx])
                # z_target, mu, logvar = modules['posterior'](h_target)
                _, z_target, _ = modules['posterior'](h_target)
            else:
                # fill tensor with normal sample (mean,std)
                z_target = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()

            # decode and output x at timestep (t) based on h and z
            if frame_idx < args.n_past:
                # necessary, for lstm hidden state
                modules['frame_predictor'](torch.concat([h, z_target, cond[frame_idx - 1]], dim=1))
                # update x_in for next frame prediction
                x_in = x[frame_idx]
                gen_seq.append(x_in)
            else:
                lstm_in = torch.concat([h, z_target, cond[frame_idx - 1]], dim=1)
                lstm_out = modules['frame_predictor'](lstm_in)
                x_in = modules['decoder']((lstm_out, skip))
                gen_seq.append(x_in)

        gen_seq = torch.stack(gen_seq)
        return gen_seq


def pred_rec(validate_seq, validate_cond, modules, args, device):
    """batch prediction for validation"""

    """data to device (sequence (x) and condition (cond))"""
    x = validate_seq.to(device)
    cond = validate_cond.to(device)

    """generate valid prediction"""
    with torch.no_grad():
        # initialize the hidden state.
        modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
        modules['posterior'].hidden = modules['posterior'].init_hidden()

        """predict 1~12 frames"""
        # initial input image x_in = x[0]
        gen_seq = []
        gen_seq.append(x[0])
        x_in = x[0]
        for frame_idx in range(1, args.n_past + args.n_future):
            # encode input image h at timestep (t-1) => (frame_idx-1)
            if args.last_frame_skip or (frame_idx < args.n_past):
                h, skip =  modules['encoder'](x_in)
            else:
                h, _  = modules['encoder'](x_in)

            # get latent code z at timestep (t) => (frame_idx)
            h_target, _ = modules['encoder'](x[frame_idx])
            # z_target, _, _ = modules['posterior'](h_target)
            _, z_target, _ = modules['posterior'](h_target)

            # decode and output x at timestep (t) based on h and z
            if frame_idx < args.n_past:
                # necessary
                modules['frame_predictor'](torch.concat([h, z_target, cond[frame_idx-1]], dim=1))
                # update x_in for next frame prediction
                x_in = x[frame_idx]
                gen_seq.append(x_in)
            else:
                lstm_in = torch.concat([h, z_target, cond[frame_idx - 1]], dim=1)
                lstm_out = modules['frame_predictor'](lstm_in)
                x_in = modules['decoder']((lstm_out, skip))
                gen_seq.append(x_in)

        gen_seq = torch.stack(gen_seq)
        return gen_seq
