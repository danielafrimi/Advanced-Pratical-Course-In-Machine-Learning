import os
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
from skimage.util import view_as_windows as viewW

EPSILON = 0.001
local_dir_path, f_name = os.path.split(__file__)
PLOTS_SAVE_DIR = os.path.join(local_dir_path, "plots")


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

    start = time()
    denoised_patches = denoise_function(noisy_patches, model, noise_std)
    end = time()

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
    h, w, = np.shape(image)
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

    fig = plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')

    fig.savefig(PLOTS_SAVE_DIR)
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
    start = time()
    cov = np.cov(X)
    mean = np.mean(X, axis=1)
    end = time()
    print("MVN training time is: %s" % (end-start))
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
    start = time()
    cov, prob_vector = EM_algorithm_gsm(X, k)
    end = time()
    print("GSM training time is: %s" % (end - start))
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

def weiner_formula(Y, cov, mean, noise_std):
    cov_inv = np.linalg.inv(cov)
    variance = np.power(noise_std, 2.0)
    dimension, number_of_noisy_samples = np.shape(cov_inv)
    I = np.eye(dimension, number_of_noisy_samples)

    first_expression = np.linalg.inv(cov_inv + (1 / variance) * I)
    second_expression = (cov_inv.dot(mean) + (1 / variance) * Y.T).T

    return first_expression.dot(second_expression)

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

    # Weiner formula for denosing the images
    return weiner_formula(Y, mvn_model.cov, mvn_model.mean, noise_std)

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

    cov = np.copy(gsm_model.cov)
    k = len(cov)
    D, M = Y.shape

    # Create K eye matrix for each guassian
    I = np.full([k, * cov.shape[1:]], fill_value=np.eye(cov.shape[1], cov.shape[2]))
    noise_mat = noise_std * I

    c = calculte_posterior_probability(Y, gsm_model.mix, cov + noise_mat)

    # Multiply each weiner (that fits to the cov of the i'th guassian) with th coresponded coefficients
    result = np.array([c[:, i] * weiner_formula(Y=Y, cov=cov[i], mean=np.zeros([D]), noise_std=noise_std) for i in range(k)])
    return np.sum(result, axis=0)


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
    Calcultes the posterior probability

    :param X: The patches of the images
    :param prob_vector_each_guassian: the prob that sample X came from Y_i gaussain
    :param cov_arr: K Covariance Matrixes (for each guassian)
    :return: the posterior probability
    """

    number_of_gaussians = len(cov_arr)
    number_samples = X.shape[-1]

    C = np.array([multivariate_normal.logpdf(X.T, cov=cov_arr[i], allow_singular=True)
                  for i in range(number_of_gaussians)]).T + np.log(prob_vector_each_guassian)

    # Working on log space (taking log and then exp), the shape is (number_samples,number_samples)
    C = np.exp(C - logsumexp(C, axis=1).reshape(number_samples, 1))
    return C

def update_GSM_guassians_scalars(X_Transpose_COV_X, C, dimension):
    """
    Updates the scalars for each covariance matrix

    :param X_Transpose_COV_X:
    :param C: the posterior probability
    :param dimension: dimension of the samples
    :return: new scalars for each covariance matrix
    """
    r_y_numerator = np.sum(X_Transpose_COV_X.dot(C), axis=0)
    r_y_denominator = dimension * np.sum(C, axis=0)
    r_i = r_y_numerator / r_y_denominator
    return r_i


def EM_algorithm_gsm(X, k):
    """
    Implemention of the  expectation–maximization (EM) algorithm.
    This algorithm finds (local) maximum likelihood  (iterative)
    :param X: The patches of the images
    :param k: Numbers of gussains
    :return: Optimized params for maximizing the likelihood (for each model we optimize other params)
    """

    # Save the likelihoods in each step of the gsm
    likelihoods_arr = np.array([])

    X = np.array(X)
    if np.ndim(X) == 2:
        dimension, number_samples = X.shape
    else:
        dimension, number_samples = 0, X.shape[0]

    # Randomize the scalars for each covariance matrix
    r_y = np.random.rand(k)

    # Probability vector for the mixture of the gaussians
    prob_vector_each_guassian = np.ones([k]) / k
    cov = np.cov(X)

    # Create K covraiance matrix with different scalars
    cov_arr = np.array([r_y[i] * cov for i in range(k)])

    # This is a constant, for calculating the new r_i, we need to calculate (X_i.T * COV^-1 * X_i)
    X_Transpose_COV_X = np.diag(X.T.dot(np.linalg.pinv(cov)).dot(X)).reshape([1, -1])

    likelihood = -np.inf
    previous_likelihood = np.inf


    # Run until there is converges
    while (abs(likelihood - previous_likelihood)) > EPSILON:
        previous_likelihood = likelihood

        # E Step
        C = calculte_posterior_probability(X, prob_vector_each_guassian, cov_arr)

        # M Step
        prob_vector_each_guassian = np.sum(C, axis=0) / number_samples

        r_i = update_GSM_guassians_scalars(X_Transpose_COV_X, C, dimension)
        cov_arr = np.array([r_i[i] * cov for i in range(k)])

        likelihood = GSM_log_likelihood(X, GSM_Model(cov_arr, prob_vector_each_guassian))
        likelihoods_arr = np.append(likelihoods_arr, likelihood)

    plot_maximum_ll(likelihoods_arr, "gsm-likelihood_k{}".format(k), k)

    return cov_arr, prob_vector_each_guassian

def create_GSM_model(patches, k) -> GSM_Model:
    """

    :param patches: the patches of the images
    :param k: Numbers of gussains
    :return: GSM Model
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

def plot_maximum_ll(maximum_ll_arr, title, k):
    """
    Saving the plot of the likelihood
    :param maximum_ll_arr: the likelihood array (in EM Algorithm)
    :param title: name of the file
    :param k: number of gusaains
    :return: None
    """
    plt.plot(maximum_ll_arr)
    plt.xlabel('Iteration')
    plt.ylabel('Likelihood')
    plt.suptitle('Likelihood Error GSM Model With k{} - EM Algorithm'.format(k))

    if not os.path.isdir(PLOTS_SAVE_DIR):
        os.makedirs(PLOTS_SAVE_DIR)

    save_path = os.path.join(PLOTS_SAVE_DIR, title)
    plt.savefig(save_path)


def main():

    os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=(8, 8))
    standard_images = grayscale_and_standardize(test_pictures, False)

    # Test for each model how it denoise the image
    _test_denoising(image=np.random.choice(standard_images), model=learn_GSM(patches, 3), denoise_function=GSM_Denoise)
    _test_denoising(image=np.random.choice(standard_images), model=learn_MVN(patches), denoise_function=MVN_Denoise)



if __name__ == '__main__':
    main()

