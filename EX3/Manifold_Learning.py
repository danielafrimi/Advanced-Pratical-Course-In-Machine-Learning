import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from sklearn import datasets
from sklearn.manifold import LocallyLinearEmbedding
from EX3.Manifold_visualizer import *
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import inv
from sklearn.decomposition import PCA


def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()

def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000, extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    N = len(X)
    H = np.eye(N) - 1 / N
    S = -0.5 * np.dot(np.dot(H, X), H)

    # S is PSD and symmetric matrix. therefore S diagonalize matrix, and we can find the eigen vectors/values of S
    eigenValues, eigenVectors = np.linalg.eig(S)

    # Order the  eigen vectors/values from the biggest to lowest
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    # Takes the first d eigvalue and the corresponded eighvectors
    Y = np.dot(eigenVectors[:, :d], np.diag(np.sqrt(eigenValues[:d])))

    # Each row i (Y_i) is the new vector in the lowest space correspond to X_i
    return Y, eigenValues


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''
    embedding = LocallyLinearEmbedding(n_neighbors=k, n_components=d)
    X_embedded = embedding.fit_transform(X)
    return X_embedded

def create_similarity_kernel(X, sigma):
    """

    :param X:
    :param sigma:
    :return:
    """

    pairwise_dists = squareform(pdist(X, 'euclidean'))
    K = np.exp(-pairwise_dists ** 2 / sigma)
    return K


def DiffusionMap(X, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    '''
    similarity_matrix = create_similarity_kernel(X, sigma)

    # Normalize th rows
    similarity_matrix /= similarity_matrix.sum(axis=1).reshape(-1, 1)

    # Diagonalize the similarity matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix)

    # Order the eigenvalues & eigenvectors in descent order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Select only the eigenvectors corresponding to the 2...(d + 1) highest eigenvalues.
    d_eigenvectors = eigenvectors[:, 1:d + 1]
    d_eigenvalues = eigenvalues[1:d + 1]

    # Return those eigenvectors, where the i-th eigenvector is multiplied by (\lambda_i)^t
    return np.power(d_eigenvalues, t) * d_eigenvectors


def rotate_in_high_dim_and_inject_noise(low_dim_data, dim=3, noise_std=0.125):
    """

    This function takes a data in low dimension and project it to high dimension
    while performing a random rotation and adding gaussian noise in the high dimension.

    :param low_dim_data: The low dimensional data.
    :param dim: The high dimension to use.
    :param noise_std: Standard deviation of the gaussian noise to add in the high dimension.

    :return: The high dimensional data and the rotation matrix that was used.
    """

    N, low_dim = low_dim_data.shape

    # Pad each low vector with zeros to obtain a 'dim'-dimensional vectors.
    padded_data = np.pad(array=low_dim_data,
                         pad_width=((0, 0), (0, dim - low_dim)),
                         mode='constant')

    # Random square matrix
    gaussian_matrix = np.random.rand(dim, dim)

    # Q is Ortogonal matrix -> it means this is a rotation matrix. 'every square matrix can decompose of QR'
    rotation_matrix, _ = np.linalg.qr(gaussian_matrix)

    # Rotate the padded vectors using the random generated rotation matrix.
    rotated_data_high_dim = np.dot(padded_data, rotation_matrix)

    # Add noise
    rotated_data_high_dim += np.random.normal(loc=0, scale=noise_std,
                                                 size=rotated_data_high_dim.shape)

    return rotated_data_high_dim, rotation_matrix



def get_gaussians_2d(k=8, n=128, std=0.05):
    """
    Generate a synthetic dataset containing k gaussians,
    where each one is centered on the unit circle
    (and the distance on the sphere between each center is the same).
    Each gaussian has a standard deviation std, and contains n points.

    :param k: The amount of gaussians to create
    :param std: The standard deviation of each gaussian
    :param n: The number of points per gaussian
    :returns: An array of shape (N, 2) where each row contains a 2D point in the dataset.
    """

    angles = np.linspace(start=0, stop=2 * np.pi, num=k, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Create an empty array that will contain the generated points.
    points = np.empty(shape=(k * n, 2), dtype=np.float64)

    # For each one of the k centers, generate the points by sampling from a normal distribution in each axis.
    for i in range(k):
        points[i * n: i * n + n, 0] = np.random.normal(loc=centers[i, 0], scale=std, size=n)
        points[i * n: i * n + n, 1] = np.random.normal(loc=centers[i, 1], scale=std, size=n)

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=5)
    plt.show()

    return points


def scree_plot(noise_std_arr=None, high_dim=128):
    """

    :param noise_std_arr:
    :param high_dim:
    :return:
    """

    if noise_std_arr is None:
        noise_std_arr = [0, 0.05, 0.1, 0.13, 0.2, 0.3, 0.4]

    low_dim_dataset = get_gaussians_2d()

    for noise_std in noise_std_arr:
        high_dim_noise_injected, _ = rotate_in_high_dim_and_inject_noise(low_dim_dataset, dim=high_dim,
                                                                         noise_std=noise_std)

        embedded_data_mds, eigenvalues = MDS(distance_matrix(high_dim_noise_injected, high_dim_noise_injected), d=2)

        plot_embedded_data(high_dim_noise_injected, embedded_data_mds, title="MDS Embedded data with noise {}".format(noise_std))

        plot_eigenvalues(eigenvalues, number_eigenvalues=10,
                         title=f'MDS_distance_matrix_eigenvalues_with_noise_{noise_std:.2f}')


if __name__ == '__main__':
    # MDS_swiss_roll()
    # LLE_swiss_roll()
    # diffusion_map_swiss_roll()
    # scree_plot()
    faces_embeddings()

