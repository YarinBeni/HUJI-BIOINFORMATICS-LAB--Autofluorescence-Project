import torchvision
import matplotlib.pyplot as plt
from pandas.core.common import flatten
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import glob

# #####################################################################################################################
# Folder Structure:
#####################################################################################################################

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
params = {
    # optional parameters maybe more ?    "model": "U-net","device": "cuda","lr": 0.001,
    "batch_size": 3,  # todo: at the moment the number of image per sample check if need to be changed
    "num_workers": 4,  # todo: relearn whats workers and ask what will be ours
    "image_size": 256,  # todo: to ask if this need to be the max from worms size padded up square or rectangle and how
    "in_channels": 1,  # TODO: to make sure because its gray image 1 channel or maybe 3 from tensor size?
    "num_classes": 0  # TODO: to make sure we dont have classes because our label is an image
}
#######################################################
#               Define Transforms
#######################################################
# todo: when i preprocessed the raw-images i used imread defualt fiture that was color image and now my database images
#  1)do I need to preprocess them again with flag for gray or we can work from here?
#  2)do I need to add to train transform Compose the function Grayscale(num_output_channels=1) or is imread() with
#  flag 0 in __Getitem__ is enough?
train_transform = transforms.Compose([transforms.ToTensor()])

####################################################
#       Create Train, Valid and Test sets
####################################################

# todo: how to split dataset into train and test sets? in folders structure in advanced or in runtime?
# todo: to ask if the label of each sample will be additional image in the sample folder for each sample?
# yarin solution:
# maybe to get labels folder path parameter and make a folder of labels(images) with name according to sample name and
# use make_path_list of labels(images) and to make image_to_label dictionary and make WormsDataset get the the
# dictionary as parameter and than to make get__item return (image, label=dic[images_path[index])?


TRAIN_DATASET_PATH = r"C:\Users\yarin\PycharmProjects\pythonProject\tempo_dataset\database_second_iter"


# test_data_path = 'images/test' if folders are pre-divided to test and train folders

# this run on database folders and make a list of images paths when

def make_paths_list(data_path):
    """function run on the database folder and make a list of the images when
    if n is the fixed number of images for every sample from the k samples
    than for every sample number 1<i=<k the index (i-1)*n=<j<=i*n-1 is for the same sample
    example: if k=5 and n=3 than for 2=i sample the images index are 3=<j=<5"""
    path_list = []
    for data_path in glob.glob(data_path + "/*"):
        path_list.append(glob.glob(data_path + "/*"))
    path_list = list(flatten(path_list))
    return path_list


train_image_paths = make_paths_list(TRAIN_DATASET_PATH)


# todo: ask if need to shuffle the paths of images
# random.shuffle(train_image_paths)

def show_dataset_paths(paths):
    """this 4 lines is to see whats the len and paths of the dataset."""
    n = 3  # n is the fixed number of images per sample
    print("this is number of paths of images in the dataset: \n", len(paths))
    for i in range(len(paths)):
        print('train_image_path index:{}: \n'.format(i), paths[i])
        if (i + 1) % n == 0:  # separate each sample
            print()


# if want to print delete and UN-comma: show_dataset_paths(train_image_paths)


# from here until row 110 are things mayby will be in use according to solution to the to do up top.

# split train valid from train paths (80,20) in runtime train_image_paths, valid_image_paths = train_image_paths[
# :int(0.8 * len(train_image_paths))], train_image_paths[int(0.8 * len(train_image_paths)):]

# this is to do the same for test folder if folders are pre-divided to test and train folders
# test_image_paths = []
# for data_path in glob.glob(test_data_path + '/'):
#     test_image_paths.append(glob.glob(data_path + '/'))
# test_image_paths = list(flatten(test_image_paths))


class WormsDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):  # todo: ask if its ok i am running over database class __getitem__ func?
        image_filepath = self.image_paths[index]

        # todo: ask why cv2.imread() not referenced? saw pycharm error, and its working ! maybe ignore
        image = cv2.imread(image_filepath, 0)  # 0 is grey flag

        # todo: ask if any transformation to the image is needed?
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        y_label = image  # todo: temporary until labels structure is solved, than change it to real label
        return image, y_label


#######################################################
#                  Create Dataset
#######################################################


train_dataset = WormsDataset(train_image_paths, train_transform)


# valid_dataset = WormsDataset(valid_image_paths,test_transforms) #test transforms are applied
# test_dataset = WormsDataset(test_image_paths,test_transforms)


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
        image, label = sample
        plt.imshow(image.squeeze(), cmap="gray")
        plt.show()
    #     cnt += 1
    # print("this is number of images showed :",cnt)
    plt.close()


# todo: need to change to show only one grid with next iter or to fix a batch[0] in end of epoch to show
def show_in_grid(images_iter):
    """gets dataloader object and print in one grid all images of same sample"""
    for batch in images_iter:
        # for batch in dataloader:
        # batch = next(iter(dataloader))
        images, labels = batch
        grid = torchvision.utils.make_grid(images, nrow=3)  # padding=0,nrow and padding variables need to be adjust
        plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()


# UN COMMENT the action you desired:

# #this will show in one grid all 3 samples of same image
# show_in_grid(train_loader)
# #this will show all images one by one
# show_images_one_by_one(train_dataset)


#######################################################
#                  Define Dataloader
#######################################################
# todo: until images arent same size cant use shuffle
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False)

# valid_loader = DataLoader(
#     valid_dataset, batch_size=params["batch_size"], shuffle=True
# )
#
# test_loader = DataLoader(
#     test_dataset, batch_size=params["batch_size"], shuffle=False
# )


