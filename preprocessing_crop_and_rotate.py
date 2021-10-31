import cv2
import os
import numpy as np
import imutils
#this is a change 4
I = 0
prepro = True
samples = []
coords = []  # coordinates list
cropping = False
image = []
images = []
flag = True
angle = 0
cnt = 0
DATASET_PATH = r"C:\Users\yarin\PycharmProjects\pythonProject\temp_dataset_divded"


# todo: make corp and rotated multiple times in a loop
# todo: to fix not corping from bottoms up
# todo: fix "r" for restrat rectangle in corp


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global coords, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        coords.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(image, coords[0], coords[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


def get_corp_cords(img):
    clone = img.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:  # this loop finds the cord for corp and store in refpt
        # display the image and wait for a keypress
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region without showing new rectangle its a BUG because
        # click_and_crop doesnt show restarted image
        if key == ord("r"):
            img = clone.copy()
            cv2.destroyAllWindows()
            continue
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            cv2.destroyAllWindows()
            break


def corp_image(img):
    global coords
    if coords[1][0] < coords[0][0]:  # because click_and_crop append so its switching places
        cropped_image = img[coords[0][1]:coords[1][1], coords[1][0]:coords[0][0]]
    else:
        cropped_image = img[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
    return cropped_image


def get_angle(img):
    global angle
    for ang in np.arange(0, 360*10, 5):
        rotated = imutils.rotate_bound(img, ang)
        cv2.imshow("Rotated (Correct)", rotated)
        key = cv2.waitKey(750)
        if key == ord("c"):
            cv2.imshow("Rotated (Correct)", rotated)
            key = cv2.waitKey(0)
            if key == ord("c"):
                angle = ang
                break
            if key == ord("r"):
                continue
    cv2.destroyAllWindows()


for dirpath, dirnames, filesname in os.walk(DATASET_PATH):  # search in database folder
    if filesname:  # means there is photos in the folder
        for f in filesname:  # for all pictures in sub folder
            # open new folder for the new pictures

            if f.endswith("(DIC)_M0000_ORG.tif"):  # if its the color image
                image = cv2.imread("{}\{}".format(dirpath, f))  # upload the image
                clone = image.copy()
                get_corp_cords(image)  # find cords for corp and store in refpt
                image = corp_image(clone)  # make new cropped image
                get_angle(image)  # display cropped and get rotation angle
                # open new folder for the new pictures
                dirc = r"C:\Users\yarin\PycharmProjects\pythonProject\tempo_dataset\{}".format(cnt)
                if not os.path.exists(dirc):
                    os.mkdir(dirc)
                    cnt += 1  # name of the next folder

                if len(coords) == 2:  # if there are two reference points
                    for photo_name in filesname:  # corp 3 images according to REFPT and rotate according ANGLE
                        sample = cv2.imread("{}\{}".format(dirpath, photo_name))  # read image im folder
                        clone = sample.copy()
                        roi_sample = imutils.rotate_bound(corp_image(clone), angle)  # make new cropp and rotate
                        fn, fext = os.path.splitext(photo_name)  # if in future want to change format
                        cv2.imwrite(dirc + "\{}{}".format(fn, fext), roi_sample)  # save new picture in new folder
                    coords = []  # reset refpt


############################################################################################################
#  todo: make a get pad from each image in preprocessing and than make all images same max size with padding
#                                         WORK IN PROGRESS IN THIS SECTION
############################################################################################################
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

# max_width, max_height = pad_size
# The needed padding is the difference between the
# max width/height and the image's actual width/height.
# if this work i should have a list of the padded images with the same size

# images_iter = [F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)]) for img in dataloader]
#####################################################################################################################
