import cv2
import torchvision
import numpy as np
from os import path

dataset = torchvision.datasets.MNIST('./data/',
                                     transform=torchvision.transforms.ToTensor(),
                                     download=True)

def generate_png_file(dataset, outdir:str='.'):
    for i in range(0, 100):
        img = dataset[i][0] * 255
        npimg = np.transpose(img.numpy(), (1, 2, 0))
        outfile = path.join(outdir, 'img{i:04d}.png'.format(i=i))
        cv2.imwrite(outfile, npimg)
        print(outfile)

generate_png_file(dataset)
