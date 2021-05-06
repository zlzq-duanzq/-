import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import torch.nn.functional as F

def ab2idx(img_ab):
	idx = (img_ab / 32).astype(int)
	idx = idx[0,:] * 8 + idx[1,:]
	return idx

def idx2ab(idx):
	x = (idx / 8).astype(int)
	y = idx % 8
	out = np.dstack((x,y))
	out = out * 32 + 16
	return out

def combine(img_gray, output_ab):
	output_ab = np.squeeze(np.asarray(output_ab))
	img_gray = np.squeeze(np.asarray(img_gray.cpu()))
	img_gray = img_gray * 100
	output_ab = output_ab - 128
	if img_gray.shape != output_ab.shape[0:2]:
	    output_ab = torch.from_numpy(output_ab)
	    output_ab = F.interpolate(output_ab, size=img_gray.shape, mode='bilinear')
	    output_ab = np.asarray(output_ab)
	output = np.dstack((img_gray, output_ab)).astype(np.float64)
	output_image = lab2rgb(output)
	return output_image
