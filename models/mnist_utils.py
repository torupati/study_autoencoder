import cv2
import torchvision
import numpy as np

dataset = torchvision.datasets.MNIST('./data/',
                                     transform=torchvision.transforms.ToTensor(),
                                     download=True)

for i in range(0, 100):
    img = dataset[i][0] * 255
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    print(npimg.shape)
    outfile = 'img{i:04d}.png'.format(i=i)
    cv2.imwrite(outfile, npimg)
#Image.fromarray(npimg, mode=mode)
