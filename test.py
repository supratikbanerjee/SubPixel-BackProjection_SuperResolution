import argparse
import time
import os

import torch
import yaml

import util
from data import dataset
from models import create_model


def main(config):
	device = torch.device(config['device'])

	##### Setup Dirs #####
	experiment_dir = config['path']['experiments'] + config['name']
	util.mkdir_and_rename(
                experiment_dir)  # rename experiment folder if exists
	util.mkdirs((experiment_dir+'/sr_images', experiment_dir+'/lr_images'))

	##### Setup Logger #####
	logger = util.Logger('test', experiment_dir, 'test_' + config['name'])

	##### print Experiment Config
	logger.log(util.dict2str(config))
	
	###### Load Dataset #####
	testing_data_loader = dataset.get_test_sets(config['dataset'], logger)

	trainer = create_model(config, logger)
	trainer.print_network_params(logger)

	total_avg_psnr = 0.0
	total_avg_ssim = 0.0

	for name, test_set in testing_data_loader.items():
		logger.log('Testing Dataset {:s}'.format(name))
		valid_start_time = time.time()
		avg_psnr = 0.0
		avg_ssim = 0.0
		idx = 0
		for i, batch in enumerate(test_set):
			idx += 1
			img_name = batch[2][0][batch[2][0].rindex('/')+1:]
			# print(img_name)
			img_name = img_name[:img_name.index('.')]
			img_dir_sr = experiment_dir+'/sr_images'
			img_dir_lr = experiment_dir+'/lr_images'
			util.mkdir(img_dir_sr)
			infer_time = trainer.test(batch)
			visuals = trainer.get_current_visuals()
			lr_img = util.tensor2img(visuals['LR'])
			sr_img = util.tensor2img(visuals['SR'])  # uint8
			gt_img = util.tensor2img(visuals['HR'])  # uint8
			save_sr_img_path = os.path.join(img_dir_sr, '{:s}.png'.format(img_name))
			save_lr_img_path = os.path.join(img_dir_lr, '{:s}.png'.format(img_name))
			util.save_img(lr_img, save_lr_img_path)
			util.save_img(sr_img, save_sr_img_path)
			crop_size = config['dataset']['scale']
			psnr, ssim = util.calc_metrics(sr_img, gt_img, crop_size)
			#logger.log('[ Image: {:s}  PSNR: {:.4f} SSIM: {:.4f} Inference Time: {:.8f}]'.format(img_name, psnr, ssim, infer_time))
			avg_psnr += psnr
			avg_ssim += ssim
		avg_psnr = avg_psnr / idx
		avg_ssim = avg_ssim / idx
		valid_t = time.time() - valid_start_time
		logger.log('[ Set: {:s} Time:{:.3f}] PSNR: {:.2f} SSIM {:.4f}'.format(name, valid_t, avg_psnr, avg_ssim))
		
		iter_start_time = time.time()

		total_avg_ssim += avg_ssim
		total_avg_psnr += avg_psnr

	total_avg_ssim /= len(testing_data_loader)
	total_avg_psnr /= len(testing_data_loader)
	
	logger.log('[ Total Average of Sets: PSNR: {:.2f} SSIM {:.4f}'.format(total_avg_psnr, total_avg_ssim))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, help='Path to config YAML file.')
	args = parser.parse_args()
	with open(args.config, 'r') as stream:
	    config = yaml.safe_load(stream)
	main(config)
