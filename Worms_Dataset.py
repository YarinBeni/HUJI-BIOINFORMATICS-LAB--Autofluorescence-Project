import math
import torchvision
import matplotlib.pyplot as plt
from pandas.core.common import flatten
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import torch

# #####################################################################################################################
# Folder Structure:
#######################################################################################################################

# k samples of n images each.
# FOLDER - contains folders for each individual sample
# SAMPLES - contains images from the same worm, n the number of pictures to whatever n you prefer just make sure each
# sample folder contains the same number of images.
# name the images with their unique date followed by an underscore and the image type (example: DIC, etc.)

# FOLDER\ SAMPLE_1 \photodate_1_phototype_1
# ...
# FOLDER\ SAMPLE_1 \photodate_1_phototype_n
# ...
# FOLDER\ SAMPLE_k \photodate_k_phototype_1
# ...
# FOLDER\ SAMPLE_k \photodate_k_phototype_n
#####################################################################################################################
# I used this site : https://github.com/UtkarshGarg-UG/Deep-Learning-Projects/blob/main/Computer-Vision/Loading-Custom-Dataset/loading_custom_dataset_images.ipynb
#####################################################################################################################


####################################################
#                Define parameters
####################################################
# todo: relearn whats workers and ask what will be ours -A: itay explained NEED TO GO DEEPER in TRAINING MODEL !
params = {
    # optional parameters maybe more ?    "model": "U-net","device": "cuda","lr": 0.001,
    "batch_size": 2,  # the number of image per sample check if need to be changed -A: FOR NOW ITS FINE !
    "num_workers": 4,
    "image_max_size": (0, 0),
    "in_channels": 1,  # to make sure because its gray image 1 channel or maybe 3 from tensor size? -A: YES
    "num_classes": None  # to make sure we dont have classes because our label is an image -A: EXACTLY !
}
# to ask if this need to be the max from worms size padded up square or rectangle and how
# -A: FOR NOW FINE in future will need to be max padded!

#######################################################
#               Define Transforms
#######################################################
TRANSFORMS_DIC = {"space_transform": None, "dic_transform": None, "flor_transform": None}

#  when i preprocessed the raw-images i used imread default fixture that was color image and now my database images
#  1)do I need to preprocess them again with flag for gray or we can work from here? - A:NEED TO DO AGAIN CHANGED PIXELS!
#  2)do I need to add to train transform Compose the function Grayscale(num_output_channels=1) or is imread() with
#  flag 0 in __Getitem__ is enough?  - A: imread with falg 0 is enough !

# space_transform = transforms.Compose([])
# TRANSFORMS_DIC["space_transform"] = space_transform

# NEW TODO: 1) whats specific transforms i need to uses when using cat(NOT ON COLOUR CHANNEL ON SPACE RELATED)

####################################################
#       Preprocessing Create Train, Valid and Test sets
####################################################

# how to split dataset into train and test sets? in folders structure in advanced or in runtime? - A: datasplit
#  (Random) meanwhile self divided with index (read on torchvision.utils.data random-split) and it happens before dataloaders
#   to ask if the label of each sample will be additional image in the sample folder for each
#   sample? ** yarin solution: maybe to get labels folder path parameter and make a folder of labels(images) with
#   name according to sample name and use make_path_list of labels(images) and to make image_to_label dictionary and
#   make WormsDataset get the the dictionary as parameter and than to make get__item return (image, label=dic[
#   images_path[index])? A: the florescence picture is the label and dic is the image already inside the database!


TRAIN_DATASET_PATH = r"C:\Users\yarin\PycharmProjects\pythonProject\tempo_dataset\database_second_iter"

# test_data_path = 'images/test' if folders are pre-divided to test and train folders

# this run on database folders and make a list of images paths when


EGFP_LABEL = "00(EGFP)_M0000_ORG.tif"
DIC_LABEL = "02(DIC)_M0000_ORG.tif"


def make_paths_list(data_path):
    """function run on the database folder and make a list of the images when
    if n is the fixed number of images for every sample from the k samples
    than for every sample number 1<i=<k the index (i-1)*n=<j<=i*n-1 is for the same sample
    example: if k=5 and n=3 than for 2=i sample the images index are 3=<j=<5"""
    path_list = []
    for data_path in glob.glob(data_path + "/*"):
        path_list.append(glob.glob(data_path + "/*"))
    mixed_images = list(flatten(path_list))
    until_label = len(list(DIC_LABEL))
    path_list = [dic_image[:-until_label] for dic_image in mixed_images if dic_image.endswith(DIC_LABEL)]
    return path_list


# ask if need to shuffle the paths of images - A: no need will be happening in the dataloader
# random.shuffle(train_image_paths)

def show_dataset_paths(paths):
    """this 4 lines is to see whats the len and paths of the dataset."""
    n = 1  # n is the fixed number of images per sample
    print("this is number of paths of images in the dataset: \n", len(paths))
    for i in range(len(paths)):
        print('train_image_path index:{}: \n'.format(i), paths[i])
        if (i + 1) % n == 0:  # separate each sample
            print()


def get_rectangle(path_list):
    h_max, w_max = 0, 0
    for path in path_list:
        dic = cv2.imread(path + DIC_LABEL, 0)
        if dic.shape[-2] > h_max:
            h_max = dic.shape[-2]
        if dic.shape[-1] > w_max:
            w_max = dic.shape[-1]
    if h_max % 2 == 0:
        h_max = h_max + 1
    if w_max % 2 == 0:
        w_max = w_max + 1
    return h_max, w_max


paths_list = make_paths_list(TRAIN_DATASET_PATH)
params["image_max_size"] = get_rectangle(paths_list)


#######################################################
#                  Create Dataset
#######################################################


# from here until row 125 are things mayby will be in use according to solution to the to do up top.

# split train valid from train paths (80,20) in runtime train_image_paths, valid_image_paths = train_image_paths[
# :int(0.8 * len(train_image_paths))], train_image_paths[int(0.8 * len(train_image_paths)):]

class WormsDataset(Dataset):
    def __init__(self, image_paths, transform_dic):
        self.image_paths = image_paths
        self.transform_dic = transform_dic

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def pad_sample(dic, flor, size_tuple):
        if size_tuple[0] == 0 and size_tuple[1] == 0:
            return dic, flor
        new_image_height, new_image_width = size_tuple
        old_image_height, old_image_width = dic.shape[-2], dic.shape[-1]
        w_pad = new_image_width - old_image_width
        h_pad = new_image_height - old_image_height

        if w_pad % 2 != 0:
            w1_pad = int(math.ceil(w_pad / 2)) - 1
            w2_pad = int(math.ceil(w_pad / 2))
        else:
            w1_pad = int(math.ceil(w_pad / 2))
            w2_pad = int(math.ceil(w_pad / 2))

        if h_pad % 2 != 0:
            h1_pad = int(math.ceil(h_pad / 2)) - 1
            h2_pad = int(math.ceil(h_pad / 2))
        else:
            h1_pad = int(math.ceil(h_pad / 2))
            h2_pad = int(math.ceil(h_pad / 2))

        pad_image = (torch.nn.ZeroPad2d((w1_pad, w2_pad, h1_pad, h2_pad)))(dic)
        pad_image = pad_image[None, :, :]
        pad_label = (torch.nn.ZeroPad2d((w1_pad, w2_pad, h1_pad, h2_pad)))(flor)
        pad_label = pad_label[None, :, :]
        return pad_image, pad_label

    def __getitem__(self, index):  # ask if its ok i am running over database class __getitem__ func? -A: its fine.
        dic_image_filepath = self.image_paths[index] + DIC_LABEL
        florescence_image_path = self.image_paths[index] + EGFP_LABEL
        # ask why cv2.imread() not referenced? saw pycharm error, and its working !-A:  Seems fine ignore it !

        dic_image = torch.from_numpy(cv2.imread(dic_image_filepath, 0))  # 0 is grey flag
        florescence_image = torch.from_numpy(cv2.imread(florescence_image_path, 0))
        # todo: tell itay its uint8 and not uint16 print((dic_image).dtype)

        # print("size of dic: ", dic_image.shape)
        # print("size of flor: ", florescence_image.shape)
        # print()

        dic_image, florescence_image = self.pad_sample(dic_image, florescence_image, params["image_max_size"])
        #
        # print("size of dic: ", dic_image.shape)
        # print("size of flor: ", florescence_image.shape)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

        if self.transform_dic["space_transform"]:
            h_joined_images = torch.cat((dic_image, florescence_image), 0)
            # print("size of joined before transform: ", h_joined_images.shape)

            h_joined_images = self.transform_dic["space_transform"](h_joined_images)
            # print("size of joined after transform: \n", h_joined_images.shape)

            dic_image = h_joined_images[0]
            florescence_image = h_joined_images[1]
            # print("size of dic after transform: ", dic_image.shape)
            # print("size of flor after transform: ", florescence_image.shape)
            # print()

        if self.transform_dic["dic_transform"]:
            dic_image = self.transform_dic["dic_transform"](dic_image)

        if self.transform_dic["flor_transform"]:
            florescence_image = self.transform_dic["flor_transform"](florescence_image)

        # NEW TODO: this label is mcher image in future will be florescence
        return dic_image, florescence_image, self.image_paths[index]


# valid_dataset = WormsDataset(valid_image_paths,test_transforms) #test transforms are applied
# test_dataset = WormsDataset(test_image_paths,test_transforms)


#######################################################
#                  Define Dataloader
#######################################################
#  until images arent same size cant use shuffle - need to be fixed int the preprocessing step ! A: FIXED

# valid_loader = DataLoader(
#     valid_dataset, batch_size=params["batch_size"], shuffle=True
# )
#
# test_loader = DataLoader(
#     test_dataset, batch_size=params["batch_size"], shuffle=False
# )


def test_batch_shape(dataset_iter):
    cnt = 0
    for batch in dataset_iter:
        print(f"this is batch number {cnt}")
        print(f"this is the type of batch: {type(batch)} and len: {len(batch)}"
              f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for i in range(len(batch)):
            print(f"this is batch[{i}] type: {type(batch[i])}")
            if i == 0:
                print(f"this is the DIC images tensor shape: {batch[i].shape}")
                print(f"\nthis is DIC images tensor::\n{batch[i]}")
                print("\n-----------------------------------------------------------------------")

            if i == 1:
                print(f"this is the EGFP images tensor shape: {batch[i].shape}")
                print(f"\nthis is EGFP images tensor:\n{batch[i]}")
                print("\n-----------------------------------------------------------------------")

            if i == 2:
                print(f"this is the tuple contain the paths to the images:\n {batch[i]}")
        print(f"\n"
              f"Finish of batch number {cnt}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n.\n.\n.\n")
        cnt = +1


# test_batch_shape(train_loader)

#######################################################
#                  Visualize Dataset
#         Images are plotted after augmentation
#######################################################

def show_images_one_by_one(dataset):
    """get the dataset object of images and print them one by one"""
    # print images one-by one from a dataset object
    # cnt = 1
    for sample in dataset:
        # print("this is iter number: ", cnt)
        # lst.append(image)
        dic_image, florescence_image, path = sample
        plt.imshow(dic_image.squeeze(), cmap="gray")
        plt.show()
        plt.imshow(florescence_image.squeeze(), cmap="gray")
        plt.show()
    #     cnt += 1
    # print("this is number of images showed :",cnt)
    plt.close()


# todo: need to change to show only one grid with next iter or to fix a batch[0] in end of epoch to show
def show_in_grid(images_iter):
    """gets dataloader object and print in one grid all images of same sample"""
    for batch in images_iter:
        # batch = next(iter(dataloader))
        # add print path and show image&label cat together
        dic_image, flor_label, path = batch
        grid = torchvision.utils.make_grid(dic_image, nrow=params["batch_size"])
        plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()


############################################################################################################
# ********************************TEST the shape of the batch: **********************************************
#Worms_Dataset TEST:
#in the database_second_iter folder: 
#5 samples, folder for each sample and in every sample 3 images: EGFP,DIC,MCHER.
#when DIC is image and EGFP is label ( not using MCHER) and max contain rectangle is (203, 933)
        
# TEST the shape of the batch:
# 1) update test_path according to database_second_iter path and update num into the wanted batch size
# 2) run test_batch_shape to get the batch shape and information print into python console
############################################################################################################

#
num = 3
test_path = r"C:\Users\yarin\PycharmProjects\pythonProject\tempo_dataset\database_second_iter"

# driver:
params["batch_size"] = num
TRAIN_DATASET_PATH = test_path
train_dataset = WormsDataset(paths_list, TRANSFORMS_DIC)
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False)
test_batch_shape(train_loader)

# show_in_grid(train_loader)

# todo: read article and implement (U-net)
# todo: get from itay channel and space transformations
# todo: preprocess with eyals data
# todo: fix preprocess to give gray image with uint16 and not 8
# todo: fix image visualisation
# NEW todo: what changes required for adjusting wormsdataset?