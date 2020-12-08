import math

from EX3.Neflix_preprocess import load_files_from_disk, save_created_files
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
# import pydiffmap
from sklearn.decomposition import TruncatedSVD
from EX3.Manifold_Learning import *


def get_means_rating_movies(mat_of_movies_and_users):
    mean_rating_per_movie = np.mean(mat_of_movies_and_users, axis=1)
    rating_histogram = np.histogram(mean_rating_per_movie)
    plt.hist(rating_histogram, bins='auto')
    plt.title("Histogram of Movies Rating")
    plt.savefig(f'./plots/rating_histogram.png')
    plt.show()

def sparse_matrix(mat_of_movies_and_users):
    no_rating_number = np.count_nonzero(mat_of_movies_and_users == 0)
    height, width = mat_of_movies_and_users.shape

    # How sparse the matrix is (percent)
    return np.round(no_rating_number/ height*width)

def stats_year_realese(df_of_movies_info):
    df_of_movies_info_years = df_of_movies_info[:, 1]
    df_of_movies_info_years = df_of_movies_info_years[:5000]

    l = len(df_of_movies_info_years)
    i = 0
    while (i < l):
        if (math.isnan(df_of_movies_info_years[i])):
            df_of_movies_info_years.pop(i)
        else:
            i += 1
        l = len(df_of_movies_info_years)

    '''Plotting the distribution of number of movie released in the years'''
    plt.figure(figsize=(12, 6))
    sns.distplot(df_of_movies_info_years, color='#8E44AD');
    plt.title('Distribution of dates of movie release', fontdict={'fontsize': 30})
    plt.savefig('Distribution_of_movie_release.png')
    plt.show()



def dimension_reduction(matrix_movies_and_users):

    svd = TruncatedSVD(n_components=10).fit(matrix_movies_and_users)
    transformed_data = svd.transform(matrix_movies_and_users)

    # MDS

    # embedded_mds, _ = MDS(distance_matrix(transformed_data, transformed_data), d=2)
    # plot_embedded_data(None, embedded_mds, "MDS_netflix")

    # LLE
    for number_neighbors in [5, 10, 15, 16, 20]:

        embedded_lle = LLE(transformed_data, 2, k=number_neighbors)
        plot_embedded_data(None, embedded_lle, "embedded_lle_netflix_neighbors_{}".format(number_neighbors))



def scree_plot_data(matrix_movies_and_users):
    plt.plot(np.cumsum(matrix_movies_and_users.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')


    _, data_dimension = matrix_movies_and_users.shape

    for dimension_percent in [10, 20, 40, 60, 80]:
        dimension = np.round((dimension_percent * data_dimension) / 100)
        embedded_movies_and_users, eigenvalues = MDS(distance_matrix(matrix_movies_and_users, matrix_movies_and_users),
                                                     d=dimension)

        plot_eigenvalues(eigenvalues, number_eigenvalues=20, title="netflix_mds_{}percent".format(dimension_percent))

def save_partly_movies_user(df_of_movies_info, matrix_movies_and_users):

    cut_matrix = matrix_movies_and_users[:2000]
    cut_matrix = cut_matrix[:, :3000]
    save_created_files(df_of_movies_info, cut_matrix, "reducted_matrix_movies_and_users", "info")

def get_top_movies_users(matrix_movies_and_users, df_of_movies_info):

    # Takes the movies with the most rating
    # sum_rating_per_movie = np.sum(matrix_movies_and_users, axis=1)
    # sort_indices = np.argsort(sum_rating_per_movie, axis=0)[::-1]
    # sort_indices_sliced = sort_indices[:2000]
    # sort_indices_sliced = sort_indices_sliced.flatten().tolist()[0]
    #
    # reducted_matrix_movies_and_users = matrix_movies_and_users[sort_indices_sliced]

    # Takes the user that rated at most
    sum_rating_per_user = np.sum(matrix_movies_and_users, axis=0)
    sort_indices = np.argsort(sum_rating_per_user, axis=1)[::-1]
    sort_indices_sliced = sort_indices[:3000]
    sort_indices_sliced = sort_indices_sliced.flatten().tolist()[0]
    reducted_matrix_movies_and_users = matrix_movies_and_users[:, sort_indices_sliced]

    save_created_files(df_of_movies_info, reducted_matrix_movies_and_users, "reducted_matrix_movies_and_users", "info")
    return reducted_matrix_movies_and_users

def main():

    df_of_movies_info, matrix_movies_and_users = load_files_from_disk()
    # get_top_movies_users(matrix_movies_and_users, df_of_movies_info)
    dimension_reduction(matrix_movies_and_users)



if __name__ == '__main__':
    main()