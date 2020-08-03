import torchvision
import torch.nn as nn
#from models.modules.capsule.capsule_network import CapsuleNetwork
import models.modules.cnn_models as CNN_arch
import models.modules.SPBP as SPBP

# Generator
def define_G(config):
	net_config = config['network_G']
	model = net_config['model']
	if model == 'SRCNN':
		netG = CNN_arch.SRCNN(config['dataset']['scale'])

	elif model == 'SPBP':
		netG = SPBP.SPBP(in_channels=net_config['in_channels'], out_channels=net_config['out_channels'],
                                  num_features=net_config['num_features'], num_groups=net_config['num_groups'],
                                  upscale_factor=config['dataset']['scale'])
	return netG

def loss_criterion(loss):
	if loss == 'l1':
		criterion = nn.L1Loss()
	elif loss == 'l2':
		criterion = nn.MSELoss()

	else:
		raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss))
	return criterion