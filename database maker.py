import torch
import torchvision
import torchvision.transforms as transforms
# import os
import numpy as np
import matplotlib.pyplot as plt
# import cv2 as cv
import torch.utils.data as data
import torchvision.transforms.functional as F

BATCH_SIZE = 3  # todo: ask if needed to change (modify to 2 if sample is 2 images)
PIXEL_SIZE = 224  # todo: ask if needed to change
# mean = [] # todo: ask if needed to get mean of dataset?
# std = [] # todo: ask if needed to get std of dataset?
TRAIN_DATASET_PATH = r"C:\Users\yarin\PycharmProjects\pythonProject\tempo_dataset\database_second_iter"
TEST_DATASET_PATH = ""  # todo: make a path for test dataset

# todo: to ask if needed customize transform if so to modified and use this section
# train_transform = transforms.Compose([transforms.Resize((PIXEL_SIZE,PIXEL_SIZE)),transforms.ToTensor(),
# transforms.Normalize(torch.Tensor(mean),std)])
train_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

# test_transform = transforms.Compose([transforms.Resize((PIXEL_SIZE,PIXEL_SIZE)),transforms.ToTensor(),

# transforms.Normalize(torch.Tensor(mean),std)])

# todo: ask how to split test and train data set
# test_dataset = torchvision.datasets.ImageFolder(root = TEST_DATASET_PATH,transform=test_transform)

train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATASET_PATH, transform=train_transform)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(len(dataloader))
cnt = 1
lst = []
while cnt < 4:
    print("this is cnt:", cnt)
    images, labels = next(iter(dataloader))
    print("this is images len : ", len(images))
    print("this is images size: ", images.size())

    print("this is images[0] size: ", images[0].size())
    # print("this is images[0]: ")
    # print(images[0])
    print()
    cntt = 1
    print()
    print()
    print()
    lst.append(id(images[0]))
    images, labels = next(iter(dataloader))
    for data in images[0]:
        print("this is data size", data.size())
        print("this is data type: ", type(data))
        print("this is data from images, picture number: ", cntt)
        cntt += 1
        print()
        # print(data)
    print("#############################")
    print()
    cnt += 1
x = lst[0]
for a in lst[1:]:
    if a == x:
        print("same")
print("not same")




# visualization of the pictures
torch.set_printoptions(linewidth=120)
len(train_dataset)
len(dataloader)
lst = []


def show_images_onebyone(dataset):
    # print images one-by one from a dataset object
    cnt = 1
    for sample in train_dataset:
        print("this is iter number: ", cnt)
        image, label = sample
        lst.append(image)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.show()
        cnt += 1
    plt.close()


# https://stackoverflow.com/questions/61943896/pad-torch-tensors-of-different-sizes-to-be-equal
# todo: find a way to make all picture same size than make batch bigger and show in one grid, maybe different way?
def get_pad_size(dataset, pad_size=[0, 0]):
    """get the minimal contain rectangle of the a batch"""
    for sample in dataset:
        image, label = sample
        col_len = len(image[0][0])
        row_len = len(image[0])
        if col_len > pad_size[1]:
            pad_size[1] = col_len
        if row_len > pad_size[0]:
            pad_size[0] = row_len
    return pad_size


# in this data the biggest is (202,933)

pad_size = get_pad_size(train_dataset)  # maybe change to dataloader
max_width, max_height = pad_size

# The needed padding is the difference between the
# max width/height and the image's actual width/height.
image_batch = [F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)]) for img in dataloader]

# for batch in dataloader:
for batch in image_batch:
    # batch = next(iter(dataloader))
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow=15, padding=100) # nrow and padding variables need to be adjust
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()

# for batch in train_dataset:
