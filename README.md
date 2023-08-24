# study_autoencoder

To start, please prepare python 3.9.17 because this version is defined in pyproject.toml.

```
$ pyenv local 3.9.17
$ poetry env use 3.9.17
```

Then, install modules.

$ poetry install

After all modules are installed, you can run the code.

```
$ poetry run python models/vae_mnist.py
```


# References


- https://avandekleut.github.io/vae/

### programming memo (TODO)

- multi-processing
  https://pytorch.org/docs/stable/notes/multiprocessing.html

- how to save and load model:
  https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

- MNIST Dataset
  https://pytorch.org/vision/0.15/datasets.html

- docker build for poetry
  https://stackoverflow.com/questions/68756419/dockerfile-multistage-python-poetry-install

  