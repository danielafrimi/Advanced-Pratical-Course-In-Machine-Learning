from EX3.Manifold_Learning import *
from sklearn.decomposition import PCA

def create_random_cmap(length):
    # How to use it:

    # cmap = create_random_cmap(len(labels_per_colors))
    # a = ax.scatter(Y[:, 0],Y[:, 1],  c=colors, cmap=cmap)
    cmap = plt.get_cmap(np.random.choice(["Set1", "Set2", "Set3", "Dark2", "Accent"]))
    cmap.colors = cmap.colors[:length]
    cmap.N = length
    cmap._i_bad = length + 2
    cmap._i_over = length + 1
    cmap._i_under = length
    return cmap



def MDS_swiss_roll():
    """
    Embed the swiss roll data points into lower dimension using MDS algorithm
    """

    data, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)
    embedded_data_mds, _ = MDS(distance_matrix(data, data), 2)

    fig = plt.figure()
    title = 'Swiss_Roll_2d_using_MDS'
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c= color , cmap=plt.cm.Spectral)
    ax.set_title("Original Data Swiss Roll")

    ax = fig.add_subplot(212)
    ax.scatter(embedded_data_mds[:, 0], embedded_data_mds[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])

    plt.title(title)
    plt.savefig(f'./plots/{title}.png')
    # plt.show()


def LLE_swiss_roll():
    """
    Embed the swiss roll data points into lower dimension using LLE algorithm
    """

    data, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    embedded_data = LLE(data, d=2, k=12)

    fig = plt.figure()
    title = 'Swiss_Roll_2d_using_LLE'
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c= color , cmap=plt.cm.Spectral)
    ax.set_title("Original Data Swiss Roll")

    ax = fig.add_subplot(212)
    ax.scatter(embedded_data[:, 0], embedded_data[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.savefig(f'./plots/{title}.png')
    # plt.show()


def diffusion_map_swiss_roll(sigmas=None, ts=None):
    """
    Try different parameters of sigma and t to use in the DiffusionMap dimensionality
    reduction algorithm for swiss roll data set.

    # TODO change
    :param sigmas: an iterable of sigma values to try.
    :param ts: an iterable of t values to try.
    """

    data, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    if sigmas is None:
        sigmas = np.arange(1, 3, 0.5)

    if ts is None:
        ts = np.arange(1, 52, 10)

    fig = plt.figure()
    plt.suptitle("Swiss_Roll_original_3d_data")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.savefig(f'./plots/Swiss_Roll_original_3d_data.png')
    plt.show()

    for sigma in sigmas:
        for t in ts:
            reduced_dim_data = DiffusionMap(data, 2, sigma, t)

            plt.figure()
            plt.suptitle(f'Swiss_Roll_2d_using_DM_sigma_{sigma}_t_{t}')
            plt.scatter(reduced_dim_data[:, 0], reduced_dim_data[:, 1], c=color, cmap=plt.cm.Spectral)
            plt.savefig(f'./plots/Swiss_Roll_2d_using_DM_sigma_{sigma}_t_{t}.png')

    plt.show()

def plot_embedded_data(original_points, embdded_points, title):

    cmap = create_random_cmap(6)
    fig = plt.figure()

    if original_points:
        ax = fig.add_subplot(211, projection='3d')
        ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2])
        ax.set_title("Original data")

    ax = fig.add_subplot(212)
    ax.scatter(embdded_points[:, 0], embdded_points[:, 1], cmap=cmap)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.savefig(f'./plots/' + title + '.png')
    plt.show()


def faces_embeddings(path='./faces.pickle'):
    """
    Dimension reduction to faces dataset on different algorithms
    :param path: Path of the faces dataset
    """

    with open(path, 'rb') as f:
        faces = pickle.load(f)

    title = 'faces_PCA'
    faces_pca = PCA(n_components=2).fit_transform(faces)
    plot_with_images(faces_pca, faces, title)
    plt.savefig(f'./plots/{title}.png')
    plt.show()

    title = 'faces_MDS'
    faces_mds, _ = MDS(distance_matrix(faces, faces), d=2)
    plot_with_images(faces_mds, faces, title)
    plt.savefig(f'./plots/{title}.png')
    plt.show()

    title = 'faces_LLE'
    faces_mds = LLE(faces, d=2, k=12)
    plot_with_images(faces_mds, faces, title)
    plt.savefig(f'./plots/{title}.png')
    plt.show()

    faces_dm_by_params = dict()
    for sigma in [1, 2, 3, 4, 5, 6]:
        for t in [3, 5, 7, 11, 17]:
            title = f'faces_DiffusionMap_sigma_{sigma}_t_{t}'
            faces_dm_by_params[(sigma, t)] = DiffusionMap(faces, 2, sigma, t)
            plot_with_images(faces_dm_by_params[(sigma, t)], faces, title)
            plt.savefig(f'./plots/{title}.png')
            plt.show()



def plot_eigenvalues(eigenvalues, number_eigenvalues, title):
    """
    :param eigenvalues: The eigenvalues to plot.
    :param number_eigenvalues: Number of eigenvalue to plot.
    :param title: Name of the plot and filename.
    """

    plt.figure()
    plt.suptitle(title)
    plt.xticks(np.arange(1, number_eigenvalues + 1))
    plt.xlabel('eigenvalue index ordered')
    plt.ylabel('eigenvalue')
    plt.plot(np.arange(1, number_eigenvalues + 1),
             eigenvalues[:number_eigenvalues])
    plt.savefig(f'./plots/{title}.png')
    plt.show()

