import torch
import torchvision
import torchvision.transforms as transforms
# import os
# import numpy as np
import cv2
import torch.utils.data as data

BATCH_SIZE = 3  # todo: ask if needed to change (modify to 2 if sample is 2 images)
PIXEL_SIZE = 224  # todo: ask if needed to change
# mean = [] # todo: ask if needed to get mean of dataset?
# std = [] # todo: ask if needed to get std of dataset?
TRAIN_DATASET_PATH = r"C:\Users\yarin\PycharmProjects\pythonProject\tempo_dataset\database_second_iter"
TEST_DATASET_PATH = ""  # todo: make a path for test dataset

# todo: to ask if needed customize transform if so to modified and use this section
# train_transform = transforms.Compose([transforms.Resize((PIXEL_SIZE,PIXEL_SIZE)),transforms.ToTensor(),
# transforms.Normalize(torch.Tensor(mean),std)])

# test_transform = transforms.Compose([transforms.Resize((PIXEL_SIZE,PIXEL_SIZE)),transforms.ToTensor(),

# transforms.Normalize(torch.Tensor(mean),std)])

# todo: ask how to split test and train data set
# test_dataset = torchvision.datasets.ImageFolder(root = TEST_DATASET_PATH,transform=test_transform)
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATASET_PATH, transform=transforms.ToTensor())

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