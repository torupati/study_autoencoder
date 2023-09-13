from sklearn.decomposition import PCA
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# dataset
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html


# https://www.kaggle.com/code/jonathankristanto/experimenting-with-pca-on-mnist-dataset

plt.close()

dataset = datasets.load_digits()
print('keys=', dataset.keys())
print('len(images)=', len(dataset.images), ' len(targets)=', len(dataset.target))

fig, axes = plt.subplots(10, 10, figsize=(15, 15), subplot_kw={'xticks':[], 'yticks':[]},\
    gridspec_kw=dict(hspace=0.5, wspace=0.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(dataset.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(dataset.target[i])
fig.savefig('100.png')

pca = PCA().fit(dataset.data)
print(pca)

fig, ax = plt.subplots(1, 1)
ax.plot(np.cumsum(pca.explained_variance_ratio_))
ax.set_xlabel('number of components')
ax.set_ylabel('cumulative explained variance')
fig.savefig('pca_cum.png')

pca = PCA(0.95).fit(dataset.data)
reduced_data = pca.transform(dataset.data)
restored_data = pca.inverse_transform(reduced_data)
print('pca.n_components_:', pca.n_components_)
print('digits.data.shape:', dataset.data.shape)
print('reduced_data.shape:', reduced_data.shape)
print('restored_data.shape:', restored_data.shape)

fig, axes = plt.subplots(10, 10, figsize=(15, 15), subplot_kw={'xticks':[], 'yticks':[]},\
    gridspec_kw=dict(hspace=0.5, wspace=0.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(restored_data[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(dataset.target[i])
fig.savefig('pca_reconstruct_095.png')
    
#pca = PCA(n_components=.95)

#pca.fit(X_train)



from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, shuffle=False)

steps = [('pca', PCA()), ('model', svm.SVC())]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)

y_model = pipe.predict(X_test)
accuracy_score(y_test, y_model)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, dataset.data, dataset.target, cv=3)
print('scores:', scores)
print('average_score:', scores.mean())