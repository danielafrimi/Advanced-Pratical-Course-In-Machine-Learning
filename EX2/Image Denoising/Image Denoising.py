import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW

EPSILON = 0.001

def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def plot_path(patches, patch_size=(8, 8)):
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                                                                          window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


# TODO i changes the signature
def _test_denoising(image, model, denoise_function,
                    noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
                noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """

    def __init__(self, cov, mix):
        self.cov = cov
        self.mix = mix


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """

    def __init__(self, P, vars, mix):
        self.P = P
        self.vars = vars
        self.mix = mix


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_size X number_of_patches matrix of image patches.›
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """
    # Intuition is how likely is the samples are come from this disterbution.
    # The log likelihood is the number of telling us how likely it is
    return np.sum(multivariate_normal.logpdf(X.T,
                                  mean=model.mean,
                                  cov=model.cov,
                                  allow_singular=False))



def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """

    number_of_gaussians = len(model.cov)

    # We calculate the likelihood for each guassian, and then sum it all
    likelihood = np.array([np.log(model.mix[i]) + multivariate_normal.logpdf(X.T, cov=model.cov[i])
                           for i in range(number_of_gaussians)]).transpose()

    # Work in log space -> normalizing in log space using the logsumexp function, and finally taking the exponent to
    # move back from log space
    return np.sum(logsumexp(likelihood, axis=1))



def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    # TODO: YOUR CODE HERE


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """

    cov = np.cov(X)
    mean = np.mean(X, axis=1)
    return MVN_Model(mean, cov)


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """

    cov, prob_vector = EM_algorithm_gsm(X, k)
    return GSM_Model(cov, prob_vector)


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """

    # TODO: YOUR CODE HERE


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    # TODO: YOUR CODE HERE


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """

    # TODO: YOUR CODE HERE


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    # TODO: YOUR CODE HERE

def calculte_posterior_probability(X, prob_vector_each_guassian, cov_arr):
    """

    :param X:
    :param prob_vector_each_guassian:
    :param cov_arr:
    :return:
    """

    number_of_gaussians = len(cov_arr)
    number_samples = X.shape[-1]

    C = np.array([multivariate_normal.logpdf(X.T, cov=cov_arr[i], allow_singular=True)
                           for i in range(number_of_gaussians)]).transpose() + np.log(prob_vector_each_guassian)

    C = np.exp(C - logsumexp(C, axis=1).reshape(number_samples, 1))
    return C


def EM_algorithm_gsm(X, k):
    """

    :param X:
    :param k:
    :param MLL_function:
    :return:
    """

    X = np.array(X)
    dimension, number_samples = np.ndim(X)

    # Randomize the scalars for each covariance matrix
    r_i, prob_scalars = np.random.random(k), np.random.random(k)

    # Probability vector for the mixture of the gaussians
    prob_vector_each_guassian = prob_scalars / prob_scalars.sum()
    cov = np.cov(X)

    # Create K covraiance matrix with different scalars
    cov_arr = np.array([r_i[i] * cov for i in range(k)])

    # This is a constant, for calculating the new r_i, we need to calculate (X_i.T * COV^-1 * X_i)
    # TODO needs diag?
    X_Transpose_COV_X = np.diag(X.T.dot(np.linalg.pinv(cov)).dot(X)).reshape([1, -1])

    likelihood = np.inf
    previous_likelihood = -np.inf

    # Run until there is converges
    # TODO check is this is loop is inf
    while (abs(likelihood - previous_likelihood)) > EPSILON:
        previous_likelihood = likelihood
        # E Step
        C = calculte_posterior_probability(X, prob_vector_each_guassian, cov_arr)

        # M Step
        prob_vector_each_guassian = np.sum(C, axis=0) / number_samples

        r_i_numerator = np.sum(np.dot(C, X_Transpose_COV_X), axis=0)
        r_i_denominator = dimension * np.sum(C, axis=0)
        r_i = np.divide(r_i_numerator, r_i_denominator)
        cov_arr = np.array([r_i[i] * cov for i in range(k)])

        likelihood = GSM_log_likelihood(X, GSM_Model(cov_arr, prob_vector_each_guassian))

    return cov_arr, prob_vector_each_guassian





def EM_algorithm(prob_vector, mu, cov, X):
    """

    :param cov:
    :param mu:
    :param prob_vector:  initial k-length probability vector for the mixture of the gaussians.
    :return:
    """
    # C_iy is a matrix size (number_of_gaussians X number_of_samples)

    number_of_guassians = len(mu)
    number_of_samples = len(X[0])
    c_i_y = np.zeros(number_of_guassians, number_of_samples)
    old_prob_vector = None

    # While not converge
    while old_prob_vector is None or prob_vector != old_prob_vector:
        old_prob_vector = prob_vector
        # Calculate the C_iy - E step

        # For the C_iy dominator we need to sum all the prob # TODO Check this
        sum_score_for_y_i_guassian = old_prob_vector * multivariate_normal.pdf(X, mean=mu, cov=cov).sum()

        for y_i in range(number_of_guassians):
            for sample_j in range(number_of_samples):
                score_for_y_i_guassian = old_prob_vector[y_i] * multivariate_normal.pdf(X[sample_j], mean=mu[y_i], cov=cov[y_i])

                c_i_y[y_i][sample_j] = np.divide(score_for_y_i_guassian, sum_score_for_y_i_guassian)

        # M Step
        for y_i in range(number_of_guassians):
            # For Calculate the prob vector of each guassian
            numbers_samples_fit_y_i = 0
            # For Calculate the mu of each guassian
            # samples_fit_y_i = 0

            for sample_j in range(number_of_samples):
                # We pass through the samples and check which sample are came from guassian y_i
                numbers_samples_fit_y_i += c_i_y[y_i][sample_j]
                # samples_fit_y_i += np.multiply(c_i_y[y_i][sample_j], X[sample_j])

            prob_vector[y_i] = np.divide(numbers_samples_fit_y_i, number_of_samples)

            # TODO for gsm we dont need to learn the mu becuase its always 0
            # mu[y_i] = np.divide(samples_fit_y_i, numbers_samples_fit_y_i)

        # TODO Calculate the Cov Matrix!!

    return prob_vector, cov


def plot_data(data):
    hx, hy, _ = plt.hist(data, bins=50, normed=1, color="lightblue")

    plt.ylim(0.0, max(hx) + 0.05)
    plt.title(r'Normal distribution $\mu_0 = 3$ and $\sigma_0 = 0.5$')
    plt.grid()

    plt.savefig("likelihood_normal_distribution_01.png", bbox_inches='tight')
    # plt.show()
    plt.close()


def create_GSM_model(patches, k):
    """

    :param patches:
    :param k: Number of the samples - we need for each sample a Covariance matrix
    :return:
    """
    cov_gsm = np.cov(patches)
    cov_scalars = np.random.rand(k)
    prob_scalars = np.random.rand(k)

    # Probability vector for the mixture of the gaussians
    prob_vector = prob_scalars / prob_scalars.sum()

    # Reshape for multiply between them
    cov_scalars = cov_scalars[:, np.newaxis, np.newaxis]
    cov_gsm_extended = np.repeat(cov_gsm[np.newaxis, :, :], k, axis=0)

    return GSM_Model(cov_gsm_extended*cov_scalars, prob_vector)

if __name__ == '__main__':
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)
    num_patches = 70
    patches = sample_patches(train_pictures, psize=(8, 8), n=num_patches)

    # Run LL for mvn & GSM model
    ll_mvn = MVN_log_likelihood(patches, MVN_Model(np.mean(patches, axis=1), np.cov(patches)))
    ll_gsm = GSM_log_likelihood(patches, create_GSM_model(patches, k=num_patches))



    # initial(patches)
    # learn_MVN(patches)

    # images_example()
