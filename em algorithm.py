import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Generate synthetic data
np.random.seed(0)
X = np.concatenate([np.random.normal(0, 1, (200, 2)), np.random.normal(4, 1, (200, 2))], axis=0)

# EM Algorithm
n_components = 2  # Number of components (clusters)
gmm = GaussianMixture(n_components=n_components)
gmm.fit(X)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap='viridis', s=40, alpha=0.6)

# Plot ellipses for each Gaussian component
ax = plt.gca()
for i in range(n_components):
    covariances = gmm.covariances_[i][:2, :2]
    mean = gmm.means_[i][:2]
    eigvals, eigvecs = np.linalg.eigh(covariances)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * np.sqrt(2) * np.sqrt(eigvals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='k', facecolor='none')
    ax.add_patch(ell)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Expectation-Maximization (EM) Algorithm")
plt.show()
