from EX3.Manifold_Learning import *



def MDS_swiss_roll():

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
    # plt.savefig(f'./plots/{title}.png')
    plt.show()


def LLE_swiss_roll():

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
    reduction algorithm.

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


    fig = plt.figure()

    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2])
    ax.set_title("Original data")

    ax = fig.add_subplot(212)
    ax.scatter(embdded_points[:, 0], embdded_points[:, 1])
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.savefig(f'./plots/' + title + '.png')
    plt.show()




def plot_faces_dataset(data_path= "EX3/faces.pickle"):
    """

    :param data_path:
    :return:
    """
    with open(data_path, 'rb') as f:
        faces = pickle.load(f)
    pass



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
