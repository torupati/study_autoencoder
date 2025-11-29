import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.mnist.dataset_mnist import get_mnist_dataset

transform = transforms.Compose(
    [
        transforms.Normalize((0.5,), (0.5,))
        # Normalize to [-1, 1] range
        # transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ]
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE.

    This module implements the core quantization mechanism that maps continuous
    encoder outputs to a discrete latent space by finding the nearest embedding vector.

    The embedding vectors form a learnable codebook (dictionary) that represents
    the discrete latent space. Each encoding from the encoder is quantized to the
    nearest embedding vector in this codebook.

    Attributes:
        n_e (int): Number of embedding vectors (codebook size)
        e_dim (int): Dimension of each embedding vector
        beta (float): Weight for the commitment loss
        embedding (nn.Embedding): Codebook containing all embedding vectors
                                  Shape: [n_e, e_dim]
                                  This is the set of vectors in the latent space.
    """

    def __init__(self, n_e, e_dim, beta):
        """
        Initialize the Vector Quantizer.

        Args:
            n_e (int): Number of embedding vectors in the codebook (e.g., 512)
            e_dim (int): Dimension of each embedding vector (e.g., 64)
            beta (float): Weight for commitment loss (typically 0.25)
        """
        super(VectorQuantizer, self).__init__()
        self.beta = beta

        # Create the codebook: a learnable embedding matrix
        # Shape: [n_e, e_dim] - This is the set of discrete vectors in the latent space
        self.embedding = nn.Embedding(n_e, e_dim)
        # Initialize embedding vectors uniformly
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    @property
    def number_of_embedding_vectors(self) -> int:
        """Read-only property: Number of embedding vectors (codebook size)."""
        return self.embedding.weight.shape[0]

    @property
    def embedding_dimension(self) -> int:
        """Read-only property: Dimension of each embedding vector."""
        return self.embedding.weight.shape[1]

    def forward(self, z):
        """
        Quantize continuous latent codes to the nearest embedding vectors.

        Args:
            z (Tensor): Continuous latent codes from encoder
                       Shape: [batch_size, e_dim, height, width]

        Returns:
            _loss (Tensor): VQ loss (commitment loss + embedding loss)
            z_q (Tensor): Quantized latent codes
                         Shape: [batch_size, e_dim, height, width]
            min_encodings (Tensor): One-hot encoded indices of nearest embeddings
            min_encoding_indices (Tensor): Indices of nearest embedding vectors

        Process:
            1. Reshape z for distance calculation
            2. Compute Euclidean distance from each z to all embedding vectors
            3. Find nearest embedding for each z element
            4. Replace z with quantized z_q (nearest embedding)
            5. Compute VQ losses to optimize both encoder and codebook
        """
        # Permute z from [B, C, H, W] to [B, H, W, C] for processing
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dimension)  # [B*H*W, C]

        # Calculate Euclidean distance between z_flattened and all embedding vectors
        # d[i,j] = ||z_i - e_j||^2
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find the index of the nearest embedding vector for each z
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.number_of_embedding_vectors
        ).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get the quantized values (nearest embedding vectors)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Compute VQ loss: commitment loss + embedding loss
        # Commitment loss: encourages encoder output to stay close to chosen embedding
        # Embedding loss: encourages embedding vectors to move towards encoder outputs
        _loss = torch.mean((z.detach() - z_q) ** 2) + self.beta * torch.mean(
            (z - z_q.detach()) ** 2
        )

        # Straight-through estimator: copy gradients from z_q to z for backprop
        z_q = z + (z_q - z).detach()

        # Permute back from [B, H, W, C] to [B, C, H, W]
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
        # output dimension of encoder: [batch_size, 128, height, width]
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # output dimension of encoder: [batch_size, embedding_dim=64, height, width]
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
