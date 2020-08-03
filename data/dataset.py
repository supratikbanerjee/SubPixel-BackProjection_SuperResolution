import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import numpy as np
import random
import os
from PIL import Image
import torch

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


class Dataset(data.Dataset):
	def __init__(self, config, phase):
		super(Dataset, self).__init__()
		image_dir = config[phase]['data_location']
		self.PATH = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image(x)]
		self.config = config
		self.phase = phase
		self.repeat = self.config[self.phase]['repeat']
		self.upscale_factor = self.config['scale']
		self.LR = []
		self.HR = []
		if self.config['read'] == 'ram':
			self.enum_dl()

	def __getitem__(self, index):

		if self.phase == 'train':
			index = index % len(self.PATH)
		path = self.PATH[index]
		if self.config['read'] == 'ram':
			LR = self.LR[index]
			HR = self.HR[index]
		else:
			HR = load_image(path)
			LR = HR.resize((int(HR.size[0]//self.upscale_factor), int(HR.size[1]//self.upscale_factor)), Image.BICUBIC)
				
			
		if self.phase == 'train':
			crop_size = self.config[self.phase]['lr_size']
			LR, HR = get_patch(LR, HR, crop_size, self.upscale_factor)
			LR, HR = augment([LR, HR])
			
		LR = img2tensor(LR, 255) 
		HR = img2tensor(HR, 255)
		return LR, HR, path


	def enum_dl(self):
		batch_size = self.config[self.phase]['batch_size']
		with tqdm(total=len(self.PATH), desc=self.config[self.phase]['name'], miniters=1) as t:
			for i, path in enumerate(self.PATH):
				img = mod_crop(load_image(path), self.upscale_factor)
				LR = img.resize((int(img.size[0]//self.upscale_factor), int(img.size[1]//self.upscale_factor)), Image.BICUBIC)
				self.LR.append(LR)
				self.HR.append(img)
				t.set_postfix_str("Batches: [%d/%d]" % ((i+1)//batch_size, len(self.PATH)/batch_size))				
				t.update()

	def __len__(self):
		return len(self.PATH) * self.repeat

def img2tensor(img, rgb_range):
	img = np.array(img)
	# if img.shape[2] == 3: # for opencv imread
	#     img = img[:, :, [2, 1, 0]]
	np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
	tensor = torch.from_numpy(np_transpose).float()
	tensor.mul_(rgb_range / 255.)

	return tensor

def get_patch(img_in, img_tar, patch_size, scale):
	ih, iw = img_in.size[:2]
	oh, ow = img_tar.size[:2]
	ip = patch_size
	tp = ip * scale
	ix = random.randrange(0, iw - ip + 1)
	iy = random.randrange(0, ih - ip + 1)
	tx, ty = scale * ix, scale * iy
	img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
	img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))
	return img_in, img_tar


def augment(imgs, hflip=True, rot=True):
	# horizontal flip OR rotate
	im_list = []
	hflip = hflip and random.random() < 0.5
	vflip = rot and random.random() < 0.5
	rot90 = rot and random.random() < 0.5
	for img in imgs:
		img = np.array(img)
		if hflip: img = img[:, ::-1, :]
		if vflip: img = img[::-1, :, :]
		if rot90: img = img.transpose(1, 0, 2)
		im_list.append(img)
	return im_list

def mod_crop(im, scale):
	h, w = im.size[:2]
	return im.crop((0,0,h - (h % scale), w - (w % scale)))

def is_image(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_image(path):
	img = Image.open(path).convert('RGB')
	return img

def get_train_set(config):

	train_set = Dataset(config, 'train')
	return data.DataLoader(dataset=train_set, batch_size=config['train']['batch_size'], shuffle=config['train']['shuffle'], pin_memory=True)
	
def get_test_set(config):

	test_set = Dataset(config, 'test')
	return data.DataLoader(dataset=test_set, batch_size=config['test']['batch_size'], shuffle=config['test']['shuffle'], pin_memory=True)

def get_test_sets(config, logger):
	test_dataloaders = {}
	for test_set in config:
		if test_set != 'scale' and test_set != 'read':
			# print(test_set)
			test_data = Dataset(config, test_set)
			test_dataloader = data.DataLoader(dataset=test_data, batch_size=config[test_set]['batch_size'], shuffle=config[test_set]['shuffle'], pin_memory=True)
			test_size = len(test_dataloader)
			logger.log('Test Images: {:,d} in {:s}'.format(test_size, config[test_set]['name']))
			test_dataloaders[config[test_set]['name']] = test_dataloader
	return test_dataloaders