import sys
import numpy as np
sys.path.append("..")
from Util import PCA
from Util import normalize, asRowMatrix, read_images
from Visual import subplot
import matplotlib.cm as cm
from Util import project, reconstruct

[X, y] = read_images("att_faces")
[D, W, mu] = PCA(asRowMatrix(X), y)
E = []
for i in xrange(min(len(X), 16)):
	e = W[:,i].reshape(X[0].shape)
	E.append(normalize(e, 0, 255))
subplot(title="Eigenfaces AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet, filename="python_pca_eigenfaces.png")
steps = [i for i in xrange(10, min(len(X), 400), 20)]
E = []
for i in xrange(len(steps)):
	numEvs = steps[i]
	P = project(W[:,0:numEvs], X[0].reshape(1, -1), mu)
	R = reconstruct(W[:,0:numEvs], P, mu)
	R = R.reshape(X[0].shape)
	E.append(normalize(R, 0, 255))
subplot(title="Reconstruction AT&T Facedatabase", images=E, rows=4, cols=5, sptitle="Eigenvectors", sptitles=[], colormap=cm.gray, filename="python_pca_recostruction.png")