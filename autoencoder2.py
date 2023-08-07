"""
https://bytepawn.com/building-a-pytorch-autoencoder-for-mnist-digits.html

https://qiita.com/fukuit/items/215ef75113d97560e599

"""

import torch; torch.manual_seed(0)
import torch.nn as nn
import torchvision as tv
from torch.autograd import  Variable
import torchvision.transforms as transforms

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, 10),
            # 10
            nn.Softmax(),
            )
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(10, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
            )
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec


num_epochs = 100
batch_size = 64

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.1307, ))])
trainset = tv.datasets.MNIST(root='./data',  train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
testset = tv.datasets.MNIST(root='./data',  train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(testset, batch_size=10*1000, shuffle=True, num_workers=4)

distance = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cpu()
        output = model(img)
        loss = distance(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


confusion_matrix = np.zeros((10, 10))
for data in dataloader:
    img, labels = data[0].to(device), data[1].to(device) 
    imgs = Variable(imgs).cpu()
    encs = model.encoder(imgs).detach().numpy()
    for i in range(len(encs)):
        predicted = np.argmax(encs[i])
        actual = labels[i]
        confusion_matrix[actual][predicted] += 1
