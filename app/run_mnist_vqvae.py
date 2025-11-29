import matplotlib.pyplot as plt
import models.mnist.vqvae as vqvae_module
import torch
from models.mnist.vqvae import VQVAE
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.mnist.dataset_mnist import get_mnist_dataset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set device for vqvae module
vqvae_module.device = device

transform = transforms.Compose(
    [
        transforms.Normalize((0.5,), (0.5,))
        # Normalize to [-1, 1] range
        # transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ]
)

# Prepare MNIST dataset using custom implementation
trainset = get_mnist_dataset(root="./data/MNIST", train=True, transform=transform)
testset = get_mnist_dataset(root="./data/MNIST", train=False, transform=transform)

# define batch size and create data loaders
batch_size = 256
trainloader = DataLoader(
    trainset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0
)
testloader = DataLoader(
    testset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0
)


epoch = 0
max_epoch = 5

print("Start Training VQ-VAE Model")
model = VQVAE(128, 32, 2, 512, 64, 0.25).to(device)

# define optimization algorithm
opt = optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.9))

# lists to record training and test losses
train_loss_log = []
test_loss_log = []


for i in range(epoch, max_epoch):
    print(f"Epoch {i}/{max_epoch}")
    train_loss: float = 0.0
    test_loss: float = 0.0
    model = model.to(device)

    model.train()  # training mode
    for img, _ in tqdm(trainloader):
        img = img.to(device, dtype=torch.float)
        opt.zero_grad()
        embedding_loss, x_hat = model(img)
        recon_loss = nn.MSELoss()(x_hat, img)
        loss = recon_loss + embedding_loss
        train_loss += loss.item()
        loss.backward()
        opt.step()

    model.eval()  # evaluation mode
    for img_t, _ in tqdm(testloader):
        img = img.to(device, dtype=torch.float)
        embedding_loss, x_hat = model(img)
        recon_loss = nn.MSELoss()(x_hat, img)
        loss = recon_loss + embedding_loss
        test_loss += loss.item()

    # calculate and display losses for each epoch
    dataset_size_train: int = (
        len(trainloader.dataset) if hasattr(trainloader.dataset, "__len__") else 1
    )
    dataset_size_test: int = (
        len(testloader.dataset) if hasattr(testloader.dataset, "__len__") else 1
    )
    train_loss /= float(dataset_size_train)
    test_loss /= float(dataset_size_test)
    print("epock %d train_loss: %.5f test_loss: %.5f" % (i, train_loss, test_loss))
    train_loss_log.append(train_loss)
    test_loss_log.append(test_loss)

    if i % 5 == 0:
        torch.save(
            {"param": model.to("cpu").state_dict(), "opt": opt.state_dict(), "epoch": i},
            f"VQVAE_{i}.pth",
        )

plt.suptitle("Loss")
plt.plot(train_loss_log, label="train_loss")
plt.plot(test_loss_log, label="test_loss")
plt.grid(axis="y")
plt.legend()
plt.savefig("VQVAE_loss.png")
plt.show()
