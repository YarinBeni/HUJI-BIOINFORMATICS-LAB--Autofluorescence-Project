import torch
import torch.nn as nn
import torchvision.transforms as F
from tqdm import tqdm
from evaluate import evaluate
# import for: 1. Create dataset 2. Split into train / validation partitions
from torchvision import transforms as T
import Worms_Dataset
# import for: 3. Create data loaders
from torch.utils.data import DataLoader
# import for: 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
from torch import optim
from Unet_model import UNet


###########################
# 1. Create dataset
###########################


def train_model_Unet(train_path, val_path):
    DIC_NAME = "dic_name"
    FOLDER_PATH = "folder_path"
    EGFP_NAME = "egfp_name"
    params = {"batch_size": 1, "num_workers": 4, "image_max_size": (0, 0),
              "in_channels": 1, "num_classes": 1,
              "T.CenterCrop": 256,  # todo: need to decided size of center crop 16^k for k in N
              # new added from original file params:

              "epochs": 50,
              "device": "cpu",
              'learning_rate': 0.0009,
              'val_percent': 0.1,
              'save_checkpoint': True,
              'img_scale': 0.5,
              'amp': False,
              'train_path': None, 'val_path': None}
    # 2021-12-23
    TRANSFORMS_DIC = {"space_transform": None, "dic_transform": F.Compose([F.RandomCrop(params['T.CenterCrop'])]),
                      "flor_transform": F.Compose([F.CenterCrop(params['T.CenterCrop'])])}
    params['train_path'] = train_path
    params['val_path'] = val_path
    train_path_arr = Worms_Dataset.make_paths_list(train_path)
    val_path_arr = Worms_Dataset.make_paths_list(val_path)
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    ######################################################
    # 2. Split into train / validation partitions
    ######################################################

    train_set = Worms_Dataset.WormsDataset(train_path_arr, TRANSFORMS_DIC)
    val_set = Worms_Dataset.WormsDataset(val_path_arr, TRANSFORMS_DIC)
    n_val = int(val_set.len())
    n_train = int(train_set.len())
    ###########################
    # 3. Create data loaders
    ###########################
    # explanation: if you load your samples in the Dataset on CPU and would like to push it during training to the GPU,
    # you can speed up the host to device transfer by enabling pin_memory.
    # This lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer.
    loader_args = dict(batch_size=params["batch_size"], num_workers=params["num_workers"], pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    ############################################################################################################
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    ############################################################################################################

    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    optimizer = optim.SGD(net.parameters(), lr=params['learning_rate'], weight_decay=1e-8, momentum=0.9)
    # todo: do we need scheduler and grad_scaler?, short explanation on torch.cuda.amp and on GradScaler ?
    # todo: old_goal: maximize Dice score new_goal: minimize MSE score done correctly?
    # ReduceLROnPlateau := Reduce learning rate when a metric has stopped improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # todo: how to switch loss i have criterion and dice_loss to switch both? if so what to set the parameters?
    criterion = nn.MSELoss()  # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    global_step = 0

    ###########################
    # 5. Begin training
    ###########################

    epochs = params['epochs']
    for epoch in range(params['epochs']):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # todo: here added paths and added to pbar.set_postfix line 119 its fine?
                images, true_masks, paths = batch['image'], batch['mask'], batch["path"]

                images = images.to(device=params['device'], dtype=torch.float32)
                true_masks = true_masks.to(device=params['device'],
                                           dtype=torch.float32)
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item(), 'path(batch)': batch["path"]})

        # Evaluation round
        division_step = (n_train // (10 * params["batch_size"]))
        if division_step > 0:
            if global_step % division_step == 0:
                val_score = evaluate(net, val_loader, params['device'])
                scheduler.step(val_score)


############################################################################################################
# ********************************TEST the  model train: **********************************************
# train.py TEST:
# two test datasets:
# sixty_img_dataset: 61 images split 41 in train 20 in validation different images in train and val
# two_img_dataset: 4 images split 2 in train 2 in validation same images in train and val
# chose test by picking 1 for 2-images dataset test and 0 for 61-image dataset test OR manually set the paths


# UNCOMMENT FROM HERE:
# if __name__ == '__main__':
#     test = input("press 1 for 2-images dataset test and 0 for 61-image dataset test:")
#     while test != '0' and test != '1':
#         test = input(
#             "press 1 for 2-images dataset test and 0 for 61-image dataset test:"
#             "\n ~IF YOU DONT GIVE VALID NUMBER YOUR TRAPPED HERE! SO YOU NEED TO CLOSE PROGRAM(RED SQUARE)~")
#
#     if test == '1':
#         train_path = r'C:\Users\yarin\PycharmProjects\pythonProject\two_img_dataset\train'
#         val_path = r'C:\Users\yarin\PycharmProjects\pythonProject\two_img_dataset\validation'
#     else:
#         train_path = r'C:\Users\yarin\PycharmProjects\pythonProject\sixty_img_dataset\train'
#         val_path = r'C:\Users\yarin\PycharmProjects\pythonProject\sixty_img_dataset\validation'
#
#     train_model_Unet(train_path, val_path)
