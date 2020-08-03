import time
import torch
import numpy as np
import os
import math
import random
from datetime import datetime
import visdom
from PIL import Image
from skimage import img_as_float
from skimage.color import rgb2ycbcr
from skimage.measure import compare_psnr, compare_ssim
import cv2


class Logger():
    # logger_name: Name of the logging class. Useful when creating multiple logger classes
    # root: directory to store the log file
    # phase: training or validation... used as name for lg file 
    def __init__(self, logger_name, root, phase):
        self.logger_name = logger_name
        self.log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))

    def log(self, message, screen=True, file=True):
        timestamp = get_log_timestamp()
        log = '{}: {}'.format(timestamp, message)
        if file:
            with open(self.log_file, 'a+') as file_handler:
                file_handler.write(log+'\n')
        if screen:
            print(log)


class Visualizer():
    def __init__(self, name, port=8068, use_visdom=True,use_incoming_socket=False):
        self.win_size = 192
        self.name = name
        self.plot_data = {}
        if use_visdom:
            import visdom
            self.vis = visdom.Visdom(port=port, env=self.name,
                                     use_incoming_socket=use_incoming_socket)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        for label, item in visuals.items():
            self.vis.image(
                np.transpose(item, (2, 0, 1)),
                opts=dict(title=label),
                win=label)

    # errors: dictionary of error labels and values
    def plot(self, data, epoch, display_id, ylabel='value'):
        time.sleep(0.1)
        if display_id not in self.plot_data:
            self.plot_data[display_id] = {
                'X': [],
                'Y': [],
                'legend': list(data.keys())
            }
        mdata = self.plot_data[display_id]
        mdata['X'].append(epoch)
        mdata['Y'].append(
            [data[k] for k in self.plot_data[display_id]['legend']])
        self.vis.line(
            X=np.stack([np.array(mdata['X'])] * len(mdata['legend']), 1),
            Y=np.array(self.plot_data[display_id]['Y']),
            opts={
                'title': display_id,
                'ytickmax': 1e-4,
                'legend': mdata['legend'],
                'xlabel': 'epoch',
                'ylabel': ylabel
            },
            win=(display_id))

    def save(self):
        self.vis.save([self.name])


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


####################
# helper functions
####################

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def get_log_timestamp():
    return datetime.now().strftime('%y-%m-%d-%H:%M:%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        #logger = logging.getLogger('base')
        #logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


####################
# image manipulation
####################


def tensor2img1(tensor, out_type=np.uint8, min_max=(0, 1)):
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0)) # HWC, BGR
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    img_np = img_np.clip(0, 255)
    return img_np.astype(out_type)

def tensor2img(tensor):
    array = np.transpose(quantize(tensor, 255).numpy(), (1, 2, 0)).astype(np.uint8)
    return array


def quantize(img, rgb_range):
    pixel_range = 255. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 255).round()


def save_img(img, img_path, mode='RGB'):
    img = img.squeeze()
    pimg = Image.fromarray(img, mode=mode)
    pimg.save(img_path)


def crop_boundaries(im, cs):
    if cs > 1:
        return im[cs:-cs, cs:-cs, ...]
    else:
        return im

def mod_crop(im, scale):
    h, w = im.shape[:2]
    # return im[(h % scale):, (w % scale):, ...]
    return im[:h - (h % scale), :w - (w % scale), ...]

def m_rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

####################
# metric
####################

# proSR implmentation 
# https://arxiv.org/abs/1804.02900 
# https://github.com/fperazzi/proSR

def eval_psnr_and_ssim(im1, im2, scale):
    im1_t = np.atleast_3d(img_as_float(im1))
    im2_t = np.atleast_3d(img_as_float(im2))

    if im1_t.shape[2] == 1 or im2_t.shape[2] == 1:
        im1_t = im1_t[..., 0]
        im2_t = im2_t[..., 0]

    else:
        im1_t = rgb2ycbcr(im1_t)[:, :, 0:1] / 255.0
        im2_t = rgb2ycbcr(im2_t)[:, :, 0:1] / 255.0

    if scale > 1:
        im1_t = mod_crop(im1_t, scale)
        im2_t = mod_crop(im2_t, scale)

        # NOTE conventionally, crop scale+6 pixels (EDSR, VDSR etc)
        im1_t = crop_boundaries(im1_t, int(scale))
        im2_t = crop_boundaries(im2_t, int(scale))

    psnr_val = compare_psnr(im1_t, im2_t)
    ssim_val = compare_ssim(
        im1_t,
        im2_t,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        data_range=1.0,
        K1=0.01,
        K2=0.03,
        sigma=1.5)

    return psnr_val, ssim_val


def calc_metrics(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = m_rgb2ycbcr(img1)
        im2_in = m_rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')