import numpy as np
import pdb
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from GaussianMixture import GaussianMixture


def gauss2d(mu, sigma, to_plot=False):
    w, h = 100, 100

    std = [np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1])]
    x = np.linspace(mu[0] - 3 * std[0], mu[0] + 3 * std[0], w)
    y = np.linspace(mu[1] - 3 * std[1], mu[1] + 3 * std[1], h)

    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T

    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    z = z.reshape(w, h, order='F')

    if to_plot:
        plt.contourf(x, y, z.T)
        plt.show()

    return z

def generate_dataset(N, mean, cov):
    return np.random.multivariate_normal(mean, cov, N)

                

        




MU1 = [3, 5]
SIGMA1 = np.array([[3, 1.0], [1.0, 3.0]])
MU2 = [-1, 7]
SIGMA2 = np.array([[1, 2], [2.0, 5.0]])
X1 = generate_dataset(600, MU1, SIGMA1)
X2 = generate_dataset(400, MU2, SIGMA2)
X = np.concatenate((X1, X2), axis=0)
model = GaussianMixture(2, 100, 0.0001)
model.fit(X)
print(model.weights, model.means, model.covs)
