from os import path

import cv2
import numpy as np
import torch


def plot_latent_each_digit(ax, autoencoder, dataset, title_str=""):
    """
    Plot latent variable z in plane (1st and 2nd dim by default)
    """
    _device = next(autoencoder.parameters()).device.type
    # fig.suptitle('Sample Projection on Latent Space')
    for y_digit in range(0, 10):
        data = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        # print("digit ", y_digit)
        z_vals = []
        for i, (x, y) in enumerate(data):
            x = x[y == y_digit]
            y = y[y == y_digit]
            x = torch.flatten(x, start_dim=1)
            _ret = autoencoder.encoder(x.to(_device))
            if isinstance(_ret, dict):
                z = _ret["z"]
                # print(f'{_ret.get("mu")=} {_ret.get("sigma")=}')
            else:
                z = _ret
            z = z.to("cpu").detach().numpy()
            if len(z_vals) == 0:
                z_vals = z
            else:
                z_vals = np.concatenate([z_vals, z])
            if len(z_vals) > 300:
                break
        ax.plot(z_vals[:, 0], z_vals[:, 1], ".", label=f"{y_digit}")
    ax.grid(True)
    ax.legend()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    if len(title_str) > 0:
        ax.set_title(title_str)
    return ax


def plot_latent(ax, autoencoder, dataset, num_lim=100):
    """
    Plot latent variable z in plane (1st and 2nd dim by default)
    """
    _device = next(autoencoder.parameters()).device.type
    ax.set_title("Sample Projection on Latent Space")
    im = None
    for i, (x, y) in enumerate(dataset):
        x = torch.flatten(x, start_dim=1)
        _ret = autoencoder.encoder(x.to(_device))
        if isinstance(_ret, dict):
            z = _ret["z"]
            # print(f'{_ret.get("mu")=} {_ret.get("sigma")=}')
        else:
            z = _ret
        z = z.to("cpu").detach().numpy()
        im = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
        if i > num_lim:
            break
    ax.grid(True)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    # fig.colorbar(im, ax=ax)
    return im


def plot_reconstructed(ax, autoencoder, r0=(-5, 10), r1=(-10, 5), num_img=12):
    """Plot reconstructed image x from z on grid

    Args
    ----
        ax (): axes in matplot figure.
        autoencoder (AutoEncoder or VariationalAutoEncoder): _description_
        r0 (tuple, optional): range of latent variable z1. Defaults to (-5, 10).
        r1 (tuple, optional): range of latent variable z2. Defaults to (-10, 5).
        num_img (int, optional): _description_. Defaults to 12.
    """
    w = 28
    img = np.zeros((num_img * w, num_img * w))
    _device = next(autoencoder.parameters()).device.type

    # Get latent dimension from encoder
    with torch.no_grad():
        dummy_input = torch.zeros(1, 784).to(_device)
        _ret = autoencoder.encoder(dummy_input)
        if isinstance(_ret, dict):
            z_sample = _ret["z"]
        else:
            z_sample = _ret
        latent_dim = z_sample.shape[1]

    for i, y in enumerate(np.linspace(*r1, num_img)):
        for j, x in enumerate(np.linspace(*r0, num_img)):
            # Create z with proper dimension, padding with zeros if needed
            if latent_dim == 2:
                z = torch.Tensor([[x, y]]).to(_device)
            else:
                z_values = [x, y] + [0.0] * (latent_dim - 2)
                z = torch.Tensor([z_values]).to(_device)

            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(w, w).to("cpu").detach().numpy()
            img[(num_img - 1 - i) * w : (num_img - 1 - i + 1) * w, j * w : (j + 1) * w] = x_hat
    im = ax.imshow(img, extent=[*r0, *r1])
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    return im


def generate_png_file(dataset, outdir: str = "."):
    for i in range(0, 100):
        img = dataset[i][0] * 255
        npimg = np.transpose(img.numpy(), (1, 2, 0))
        outfile = path.join(outdir, f"img{i:04d}.png")
        cv2.imwrite(outfile, npimg)
        # print(outfile)


# generate_png_file(dataset)
