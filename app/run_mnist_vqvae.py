import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


class DataSet(Dataset):
    def __init__(self, data, transform=False):
        self.X = data[0]
        self.y = data[1]
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = self.X[index].view(28, 28)
        label = self.y[index]
        if self.transform:
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
        return img, label


# define transformation function for image data normalization
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# prepare MNIST dataset
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

# Preprocess training and test data
x_train = train_dataset.data.reshape(-1, 784).float() / 255
y_train = F.one_hot(train_dataset.targets, 10).float()
x_test = test_dataset.data.reshape(-1, 784).float() / 255
y_test = F.one_hot(test_dataset.targets, 10).float()

# create train and test datasets
trainset = DataSet([x_train, y_train], transform=transform)
testset = DataSet([x_test, y_test], transform=transform)

# define batch size and create data loaders
batch_size = 256
trainloader = DataLoader(
    trainset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0
)
testloader = DataLoader(
    testset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0
)


# define residual layer class
class ResidualLayer(nn.Module):
    # define convolutional layers in constructor
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    # apply residual connection in forward pass
    def forward(self, x):
        x = x + self.res_block(x)
        return x


# define residual stack class
class ResidualStack(nn.Module):
    # stack specified number of residual layers in constructor
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers)

    # define forward pass to pass input through the stack
    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


# define encoder class
class Encoder(nn.Module):
    # define convolutional layers and residual stack in constructor
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    # define forward pass to encode images into feature vectors
    def forward(self, x):
        return self.conv_stack(x)


# define vector quantization class
class VectorQuantizer(nn.Module):
    # define embedding layer in constructor
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    # quantize input vectors to nearest embedding vectors in forward pass
    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        _loss = torch.mean((z.detach() - z_q) ** 2) + self.beta * torch.mean(
            (z - z_q.detach()) ** 2
        )
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return _loss, z_q, min_encodings, min_encoding_indices


class Decoder(nn.Module):
    # define deconvolutional layers in constructor
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 1, kernel_size=kernel, stride=stride, padding=1),
        )

    # define forward pass to decode feature vectors into images
    def forward(self, x):
        return self.inverse_conv_stack(x)


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    # define forward pass for VQ-VAE model
    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat


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
    train_loss = 0
    test_loss = 0
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
    train_loss /= len(trainloader.dataset)
    test_loss /= len(testloader.dataset)
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
