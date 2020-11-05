# create dataset
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import Tuple
import torchvision.transforms as transforms

cat_indices_dev_dataset = [27, 29, 35, 40, 44, 83, 88, 91, 94, 99, 105, 115, 130, 142, 150, 143, 157, 165, 193, 213, 218, 219,
                   227, 250, 276, 281, 284, 310, 312, 327, 342, 348, 355, 363, 371, 375, 384, 400, 401, 407, 418, 420,
                   425, 437, 440, 452, 454, 475, 482, 489, 490, 493, 500, 515, 521, 533, 536, 539, 553, 561, 566, 577,
                   590, 591, 593, 599, 602, 624]

cat_indices_train_dataset = [4,19,20,35,37,38,42,53,57,64,84,86,96,107,112,126,130,131,153,154,155,175,179,214,230,231,
                             234,244,249,256,262,265,276,280,281,286,296,314,329,349,409,421,446,451,454,459,465,475,
                             485,515,533,538,539,542,543,548,552,560,585,630,635,643,667,681,699,706,708,718,719,720,
                             721,743,760,764,767,780,793,810,816,819,821,827,828,838,860,883,885,892,907,928,929,932,
                             938,940,942,953,975,979,980,982,994,998,1023,1037,1040,1044,1045,1055,1063,1069,1072,1075,
                             1077,1084,1087,1100,1107,1130,1137,1142,1145,1157,1161,1168,1180,1182,1188,1192,1212,1220,
                             1225,1230,1240,1247,1248,1273,1275,1276,1283,1285,1286,1288,1290,1294,1300,1313,1318,1326,
                             1329,1330,1335,1357,1385,1386,1409,1411,1413,1437,1438,1476,1485,1488,1493,1539,1550,1555,
                             1564,1576,1587,1591,1597,1612,1625,1627,1659,1677,1694,1698,1704,1718,1737,1747,1755,1767,
                             1774,1798,1805,1810,1827,1835,1843,1853,1854,1856,1859,1889,1896,1921,1934,1941,1958,1963,
                             1969,1972,1976,2014,2015,2022,2027,2030,2034,2065,2080,2090,2141,2142,2167,2173,2175,2177,
                             2188,2191,2204,2212,2215,2216,2217,2225,2229,2245,2251,2254,2256,2257,2259,2269,2279,2299,
                             2302,2310,2312,2322,2326,2329,2339,2341,2346,2359,2361,2373,2374,2376,2378,2382,2383,2415,
                             2423,2440,2460,2465,2466,2468,2475,2497,2502,2513,2525,2534,2548,2549,2553,2587,2590,2591,
                             2607,2608,2611,2616,2619,2621,2622,2636,2638,2644,2652,2657,2660,2664,2680,2681,2692,2696,
                             2703,2711,2724,2725,2726,2737,2748,2757,2763,2764,2766,2777,2775,2787,2791,2820,2824,2830,
                             2843,2858,2878,2882,2886,2903,2936,2945,2946,2957,2960,2974,2983,3010,3029,3048,3051,3060,
                             3077,3082,3095,3100,3125,3127,3129,3137,3168,3171,3190,3196,3218,3256,3258,3259,3266,3271,
                             3281,3287,3289,3293,3303,3314,3323,3329,3335,3336,3344,3345,3347,3353,3359,3361,3403,3408,
                             3413,3414,3418,3429,3433,3454,3478,3480,3483,3484,3486,3489,3493,3503,3519,3524,3527,3532,
                             3538,3542,3549,3554,3555,3559,3569,3585,3591,3594,3596,3615,3617,3629,3644,3646,3649,3650,
                             3654,3666,3668,3694,3730,3731,3738,3739,3740,3753,3762,3763,3768,3773,3781,3787,3801,3803,
                             3809,3815,3834,3844,3853,3855,3861,3876,3878,3889,3898,3901,3903,3906,3916,3934,3938,3940,
                             3942,3943,3945,3947,3970,3976,3992,4012,4013,4015,4044,4048,4050,4051,4069,4073,4081,4082,
                             4094,4095,4099,4108,4118,4122,4123,4146,4151,4167,4177,4183,4185,4201,4207,4216,4223,4227,
                             4236,4237,4249,4253,4275,4281,4282,4284,4292,4293,4295,4309,4331,4344,4347,4355,4397,4400,
                             4414,4426,4427,4430,4443,4449,4474,4482,4487,4535,4536,4541,4591,4593,4605,4606,4616,4622,
                             4640,4641,4642,4671,4676,4686,4690,4707,4710,4712,4728,4730,4741,4757,4758,4763,4764,4773,
                             4776,4777,4786,4792,4794,4807,4819,4826,4827,4857,4864,4866,4873,4880,4886,4889,4893,4908,
                             4939,4941,4950,4961,4970,4976,4978,4984,4985,4991,4998,5008,5010,5035,5043,5047,5055,5066,
                             5068,5080,5084,5086,5089,5098,5105,5114,5118,5124,5133,5136,5139,5142,5144,5148,5150,5154,
                             5162,5173,5180,5185,5191,5192,5217,5221,5227,5237,5238,5248,5252,5254,5259,5269,5284,5295,
                             5297,5301,5303,5325,5331,5335,5338,5340,5341,5349,5358,5378,5390,5403,5406,5407,5411,5415,
                             5416,5422,5439,5443,5461,5468,5470,5477,5480,5489,5502,5508,5517,5545,5563,5565,5573,5575,
                             5595,5597,5602,5607,5623]


class MyDataset(Dataset):
    """ex1 dataset."""

    def __init__(self, the_list, transform=None):
        self.the_list = the_list
        self.transform = transform

    def __len__(self):
        return len(self.the_list)

    def __getitem__(self, idx):
        item = self.the_list[idx]
        if self.transform:
            item = self.transform(item)
        return item


def get_dataset_as_array(path='./data/dataset.pickle'):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

# useful for using data-loaders
def get_dataset_as_torch_dataset(path='./data/dataset.pickle'):
    dataset_as_array = get_dataset_as_array(path)
    dataset = MyDataset(dataset_as_array)
    return dataset

# for visualizations
def un_normalize_image(img):
    img = img / 2 + 0.5
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def label_names():
    return {0: 'car', 1: 'truck', 2: 'cat'}

def change_cats_label_in_dataset(dataset, dataset_name='train'):
    cats_indices = cat_indices_train_dataset

    if dataset_name == 'dev':
        cats_indices = cat_indices_dev_dataset

    for index in cats_indices:
        # Create an new Tuple with cat label (means  the number 2)
        dataset[index] = (dataset[index][0], 2)