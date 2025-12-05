# Setup

Here is my memo to use Github for remote repository.

## Key and Github

Generate key pair by ssh-keygen.

```
$ ssh-keygen  -t ed25519 -C torupa@email.com -f keyfile
```

Upload ```keyfile.pub``` to github setting of my repository.


## Configuration of SSH Agent

Edit ```~/.ssh/config``` to write host access setting.

```
Host github_aaa
  HostName github.com
  User git
  IdentityFile ~/.ssh/keyfile
  IdentitiesOnly yes
```

## Git Setting

```
git config --global user.name "name"
git config --global user.email "name@email.com"
```

## Clone repository

```
$ git clone github_aaa:torupati/study_autoencoder.git
$ git remote -v
origin  github_aaa:torupati/study_autoencoder.git (fetch)
origin  github_aaa:torupati/study_autoencoder.git (push)
```

You can change from ssh to https by using ```git config set-url```

https://docs.github.com/ja/get-started/getting-started-with-git/managing-remote-repositories

## Python Environment Management

This project now uses [uv](https://docs.astral.sh/uv/) for Python package and dependency management.

### Installing uv

Install uv using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Project Setup

1. Clone the repository (see Git setting above)
2. Navigate to the project directory
3. Set up the environment:

```bash
# Create virtual environment with Python 3.9+
make install

# Or manually:
uv venv --python 3.9
uv sync
```

### Common Commands

```bash
# Install dependencies
make install

# Install with development dependencies
make dev-install

# Run autoencoder
make run-autoencoder

# Run variational autoencoder
make run-vae

# Run tests
make test

# Format code
make format

# Show all available commands
make help
```

### Legacy: Pyenv and Poetry

My environment was originally using python 3.12.3 with Poetry.

```
$ python -V
Python 3.12.3
```

Then, I setupt as follows.

```
$ pyenv local 3.12
$ poetry update
$ poetry install
$ poetry shell
```

To setup path for python moduls, "install" is necessary.
