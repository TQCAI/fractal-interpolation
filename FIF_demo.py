import numpy as np
from fif import *
from pylab import *

u = np.array([0.0, 0.4, 0.7, 1.0])
v = np.array([0.0, 0.5, 0.2, 0.0])

U = np.vstack((u, v)).T
U = G(U, 0.1, balance=0)
X = FIF(U, 0.01, balance=1)

plot(U[:, 0], U[:, 1], '.-')
plot(X[:, 0], X[:, 1], '.-')
show()