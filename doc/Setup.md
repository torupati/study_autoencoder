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

## Pyenv and Poetry

```
$ pyenv local 3.12
$ poetry shell
$ poetry update
```