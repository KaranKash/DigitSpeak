import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

# construct list and map
m = {'e1':0,'e2':1,'e3':2,'e4':3,'e5':4,'e6':5,'e7':6,'e8':7,'e9':8,'eo':9,'ez':10,'s0':11,'s1':12,'s2':13,'s3':14,'s4':15,'s5':16,'s6':17,'s7':18,'s8':19,'s9':20}
eng = ['1','2','3','4','5','6','7','8','9','o','z']
span = ['0','1','2','3','4','5','6','7','8','9']
X = []
y = []
for d in eng:
    temp = np.loadtxt("./English/" + d + ".txt")
    for line in temp:
        X.append(line)
        y.append('e' + d)
for d in span:
    temp = np.loadtxt("./Spanish/" + d + ".txt")
    for line in temp:
        X.append(line)
        y.append('s' + d)
X = np.matrix(X)
y = np.matrix(y).T

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i,0], X[i,1], y[i,0],
                 color=plt.cm.Set1(m[y[i,0]] / 21.),
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)

# t-SNE embedding of the digit utterances
print("Computing t-SNE embedding")
tsne = manifold.TSNE()
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne, y,
               "t-SNE 2D embedding of English and Spanish digit utterances")
plt.savefig('tsne.png')
