import torch
from math import log10
from torch.autograd import Variable
import models.networks as networks
from collections import OrderedDict
from collections import defaultdict
import time

class CNN():
    def __init__(self, config):
        super(CNN, self).__init__()

        self.logs = OrderedDict()
        self.eval = {'psnr':0.0, 'ssim':0.0, 'ssim_epoch':0, 'psnr_epoch':0}
        self.config = config
        self.device = self.config['device']
        self.ensemble = self.config['ensemble']
        self.use_chop = self.config['use_chop']
        

        # define model
        self.netG = networks.define_G(self.config).to(self.device)
        print(self.netG)

        if self.config['is_train']:
            self.train_config = self.config['train']

            self.pixel_criterion = networks.loss_criterion(self.train_config['pixel_criterion']).to(self.device)

            self.pixel_weight = self.train_config['pixel_weight']

            if self.train_config['pixel_criterion'] == 'ADAPTIVE':
                params = list(self.netG.parameters()) + list(self.pixel_criterion.parameters())
            else:
                params = self.netG.parameters()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.train_config['lr_G'],
                                                    weight_decay=self.train_config['weight_decay_G'],
                                                    betas=(self.train_config['beta1_G'], self.train_config['beta2_G']))
            if self.train_config['lr_scheme'] == 'MultiStepLR':
                self.scheduler_G = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_G, 
                    self.train_config['lr_step'], 
                    self.train_config['lr_gamma'])
        else:
            self.load()

    def update_eval(self, psnr, ssim, epoch):
        if ssim > self.eval['ssim']:
            self.eval['ssim'] = ssim
            self.eval['ssim_epoch'] = epoch
        if psnr > self.eval['psnr']:
            self.eval['psnr'] = psnr
            self.eval['psnr_epoch'] = epoch

    def get_eval(self):
        return self.eval

    def get_target(self, input, target):
        return torch.empty_like(input).fill_(target)

    def train(self, batch):
        self.lr_input, self.hr_real = Variable(batch[0].to(self.device)), Variable(batch[1].to(self.device))
        self.netG.train()
        self.netG.zero_grad()
        self.hr_fake = self.netG(self.lr_input)
        pixel_loss_g = 0.0
        if self.train_config['cl_train']:
            loss_steps = [self.pixel_criterion(sr, self.hr_real)  for sr in self.hr_fake]
            for step in range(len(loss_steps)):
                pixel_loss_g += self.pixel_weight * loss_steps[step]
        else:
            pixel_loss_g = self.pixel_weight * self.pixel_criterion(self.hr_fake, self.hr_real) 
        pixel_loss_g.backward()
        self.optimizer_G.step()
        #for param_group in self.optimizer_G.param_groups:
        #       print(param_group['lr'])
        self.logs['p_G'] = pixel_loss_g.item()
        if isinstance(self.hr_fake, list): # to get visuals
                self.hr_fake = self.hr_fake[-1]

    def test(self, batch):
        self.netG.eval()
        self.lr_input, self.hr_real = Variable(batch[0].to(self.device)), Variable(batch[1].to(self.device))
        with torch.no_grad():
            model = self._overlap_crop_forward if self.use_chop else self.netG.forward
            infer_time_start = time.time()
            if self.ensemble:
                self.hr_fake = self._forward_x8(self.lr_input, model)
            else:
                self.hr_fake = model(self.lr_input)
            infer_time = time.time() - infer_time_start
            if isinstance(self.hr_fake, list):
                self.hr_fake = self.hr_fake[-1]
        if self.config['is_train']:
            valid_loss = self.pixel_weight * self.pixel_criterion(self.hr_fake, self.hr_real)
            self.logs['v_l'] = valid_loss.item()

        return infer_time
    
    def _forward_x8(self, x, forward_function):
        """
        self ensemble
        """
        def _transform(v, op):
            v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = []
        for aug in lr_list:
            sr = forward_function(aug)
            if isinstance(sr, list):
                sr_list.append(sr[-1])
            else:
                sr_list.append(sr)

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output
    
    def _overlap_crop_forward(self, x, shave=10, min_size=100000, bic=None):
        """
        chop for less memory consumption during test
        """
        n_GPUs = 1
        scale = self.config['dataset']['scale']
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if bic is not None:
            bic_h_size = h_size*scale
            bic_w_size = w_size*scale
            bic_h = h*scale
            bic_w = w*scale
            
            bic_list = [
                bic[:, :, 0:bic_h_size, 0:bic_w_size],
                bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
                bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
                bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if bic is not None:
                    bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

                sr_batch_temp = self.netG(lr_batch)

                if isinstance(sr_batch_temp, list):
                    sr_batch = sr_batch_temp[-1]
                else:
                    sr_batch = sr_batch_temp

                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self._overlap_crop_forward(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
                ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def save(self, epoch):
        model_path = self.config['logger']['path'] + self.config['name'] +'/checkpoint/'
        model_name = "{}_{}_x{}_netG.pth".format(epoch, self.config['network_G']['model'], self.config['dataset']['scale'])
        to_save = {
        'state_dict': self.netG.state_dict(),
        'net': self.config
        }
        torch.save(to_save, model_path+model_name)

    def plot_loss(self, visualizer, epoch):
        loss_g = defaultdict(list)
        loss_g['train'] = self.logs['p_G']
        loss_g['valid'] = self.logs['v_l']
        visualizer.plot(loss_g, epoch, 'Generator Loss', 'loss')

    def print_network_params(self, logger):
        param = OrderedDict()
        param['G'] = sum(map(lambda x: x.numel(), self.netG.parameters()))
        logger.log('Network G : {:s}, with parameters: [{:,d}]'.format(self.config['network_G']['model'], param['G']))
        
    def get_current_visuals(self):
        visuals = OrderedDict()
        visuals['LR'] = self.lr_input.detach()[0].float().cpu()
        visuals['SR'] = self.hr_fake.detach()[0].float().cpu()
        visuals['HR'] = self.hr_real.detach()[0].float().cpu()
        return visuals

    def update_learning_rate(self, epoch):
        if self.train_config['lr_scheme'] != 'None':
            self.scheduler_G.step(epoch)

    def get_logs(self):
        return self.logs

    def load(self):
        checkpoint = torch.load(self.config['path']['pretrain_model_G'], map_location=self.device)
        self.netG.load_state_dict(checkpoint['state_dict'])
