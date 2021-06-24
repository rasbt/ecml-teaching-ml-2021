import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch


def load_iris():
    
    df = pd.read_csv('iris.data', index_col=None, header=None)
    df.columns = ['x1', 'x2', 'x3', 'x4', 'y']

    d = {'Iris-versicolor': 1,
         'Iris-virginica': 2,
         'Iris-setosa': 0,
    }

    df['y'] = df['y'].map(d)

    # Assign features and target

    X = torch.tensor(df[['x2', 'x4']].values, dtype=torch.float)
    y = torch.tensor(df['y'].values, dtype=torch.int)

    # Shuffling & train/test split

    torch.manual_seed(123)
    shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)

    X, y = X[shuffle_idx], y[shuffle_idx]

    percent80 = int(shuffle_idx.size(0)*0.8)

    X_train, X_test = X[shuffle_idx[:percent80]], X[shuffle_idx[percent80:]]
    y_train, y_test = y[shuffle_idx[:percent80]], y[shuffle_idx[percent80:]]

    # Normalize (mean zero, unit variance)

    mu, sigma = X_train.mean(dim=0), X_train.std(dim=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    return X_train, y_train, X_test, y_test


def plot_iris(X_train, y_train, X_test, y_test):

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    ax[0].scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], marker='o')
    ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker='v')
    ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], marker='s')
    ax[1].scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], marker='o')
    ax[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], marker='v')
    ax[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], marker='s')
    plt.show()
    
    
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 'v', 's', '^', 'x')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    tensor = torch.tensor(np.array([xx1.ravel(), xx2.ravel()]).T).float()
    logits, probas = classifier.forward(tensor)
    Z = np.argmax(probas.detach().numpy(), axis=1)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)