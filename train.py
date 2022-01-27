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


def train_model_Unet():
    DIC_NAME = "dic_name"
    FOLDER_PATH = "folder_path"
    EGFP_NAME = "egfp_name"
    params = {"batch_size": 1, "num_workers": 4, "image_max_size": (0, 0),
              "in_channels": 1, "num_classes": 1,
              "T.CenterCrop": 256, # todo: need to decided size of center crop
              # new added from original file params:

              "epochs": 5,
              "device": "cpu",  # todo: need to change to cuda?
              'learning_rate': 0.001,
              'val_percent': 0.1,
              'save_checkpoint': True,
              'img_scale': 0.5,
              'amp': False}
    #
    TRANSFORMS_DIC = {"space_transform": None, "dic_transform": F.Compose([F.RandomCrop(params['T.CenterCrop'])]),
                      "flor_transform": F.Compose([F.CenterCrop(params['T.CenterCrop'])])}
    train_path = r'C:\Users\yarin\PycharmProjects\pythonProject\2021-12-23\train'
    val_path = r'C:\Users\yarin\PycharmProjects\pythonProject\2021-12-23\validation'
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
    # todo: do i need to normalize the image/label before training starts ? or Batchnorm is sufficient? if so when?
    ###########################
    # 3. Create data loaders
    ###########################
    # todo: pin memeory is needed?
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
                                           dtype=torch.float32)  # todo: dtype=torch.long is good?
                masks_pred = net(images[None])
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


if __name__ == '__main__':
    train_model_Unet()

######################################################################################################################
#                                              ORIGINAL CODE    ~ignore older todos
######################################################################################################################

# import argparse
# import logging
# import sys
# from pathlib import Path
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import wandb
# from torchvision import transforms as T
# from torch import optim
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
# import utils.Worms_Dataset
# from utils import Worms_Dataset
# from utils.dice_score import dice_loss
# from evaluate import evaluate
# from unet import UNet
#
# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')
#
#
# # todo: ask for specific direction in order to try and push the project forward in exam period
#
#
# def train_net(net,
#               device,
#               epochs: int = 5,
#               batch_size: int = 1,
#               learning_rate: float = 0.001,
#               val_percent: float = 0.1,
#               save_checkpoint: bool = True,
#               img_scale: float = 0.5,
#               amp: bool = False):
#     # 1. Create dataset
#     DIC_NAME = "dic_name"
#     FOLDER_PATH = "folder_path"
#     EGFP_NAME = "egfp_name"
#     params = {"batch_size": 2, "num_workers": 4, "image_max_size": (0, 0),
#               "in_channels": 1, "num_classes": 1,
#               "T.CenterCrop": 0  # todo: need to decided size of center crop to fit CNN
#               }
#
#     # todo: need to decided size of center crop - what size do i need to choose in order for it to get along with CNN
#     #  architecture ?
#     TRANSFORMS_DIC = {"space_transform": None, "dic_transform": T.CenterCrop
#         , "flor_transform": T.CenterCrop, "T.CenterCrop": params["T.CenterCrop"]}
#
#     train_path = r'C:\Users\yarin\PycharmProjects\pythonProject\2021-12-23\train'
#     val_path = r'C:\Users\yarin\PycharmProjects\pythonProject\2021-12-23\validation'
#
#     train_path_arr = Worms_Dataset.make_paths_list(train_path)
#     val_path_arr = Worms_Dataset.make_paths_list(val_path)
#
#     # 2. Split into train / validation partitions
#     train_set = Worms_Dataset.WormsDataset(train_path_arr, TRANSFORMS_DIC)
#     val_set = Worms_Dataset.WormsDataset(val_path_arr, TRANSFORMS_DIC)
#     n_val = int(val_set.len())
#     n_train = int(train_set.len())
#     # todo: do i need to normalize the image/label before training starts ? or Batchnorm is sufficient? if so when?
#
#     # 3. Create data loaders #todo: insert params[keys] as parameters.
#     loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
#     train_loader = DataLoader(train_set, shuffle=True, **loader_args)
#     val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
#
#     # (Initialize logging) # todo: please explain what happens here ?
#     experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
#     experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
#                                   val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
#                                   amp=amp))
#
#     logging.info(f'''Starting training:
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {learning_rate}
#         Training size:   {train_set.len()}
#         Validation size: {0}
#         Checkpoints:     {save_checkpoint}
#         Device:          {device.type}
#         Images scaling:  {img_scale}
#         Mixed Precision: {amp}
#     ''')
#
#     # todo explain what happening here ?
#     # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
#     optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
#     grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
#
#     # todo: how to switch loss i have criterion and dice_loss to switch both? if so what to set the parameters?
#     # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
#     criterion = nn.CrossEntropyLoss()
#     # criterion = nn.MSELoss(logits,EGFP)
#     global_step = 0
#
#     # 5. Begin training
#     for epoch in range(epochs):
#         net.train()
#         epoch_loss = 0
#         with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
#             for batch in train_loader:
#                 images = batch['image']
#                 true_masks = batch['mask']
#                 # todo: here i can add paths when to use them ?
#                 paths = batch["path"]
#
#                 assert images.shape[1] == net.n_channels, \
#                     f'Network has been defined with {net.n_channels} input channels, ' \
#                     f'but loaded images have {images.shape[1]} channels. Please check that ' \
#                     'the images are loaded correctly.'
#
#                 images = images.to(device=device, dtype=torch.float32)
#                 true_masks = true_masks.to(device=device, dtype=torch.long)
#
#                 with torch.cuda.amp.autocast(enabled=amp):
#                     masks_pred = net(images)
#                     # todo: how to switch the loss ?
#                     loss = criterion(masks_pred, true_masks) \
#                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
#                                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
#                                        multiclass=True)
#
#                 optimizer.zero_grad(set_to_none=True)
#                 grad_scaler.scale(loss).backward()
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()
#
#                 pbar.update(images.shape[0])
#                 global_step += 1
#                 epoch_loss += loss.item()
#                 experiment.log({
#                     'train loss': loss.item(),
#                     'step': global_step,
#                     'epoch': epoch
#                 })
#                 pbar.set_postfix(**{'loss (batch)': loss.item()})
#
#                 # Evaluation round
#                 division_step = (n_train // (10 * batch_size))
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                         histograms = {}
#                         for tag, value in net.named_parameters():
#                             tag = tag.replace('/', '.')
#                             histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                             histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
#
#                         val_score = evaluate(net, val_loader, device)
#                         scheduler.step(val_score)
#
#                         logging.info('Validation Dice score: {}'.format(val_score))
#                         experiment.log({
#                             'learning rate': optimizer.param_groups[0]['lr'],
#                             'validation Dice': val_score,
#                             'images': wandb.Image(images[0].cpu()),
#                             'masks': {
#                                 'true': wandb.Image(true_masks[0].float().cpu()),
#                                 'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
#                             },
#                             'step': global_step,
#                             'epoch': epoch,
#                             **histograms
#                         })
#
#         if save_checkpoint:
#             Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
#             torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
#             logging.info(f'Checkpoint {epoch + 1} saved!')
#
#
# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
#     parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
#     parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
#     parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
#                         help='Learning rate', dest='lr')
#     parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
#     parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
#     parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
#                         help='Percent of the data that is used as validation (0-100)')
#     parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
#
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     args = get_args()
#
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Using device {device}')
#
#     # Change here to adapt to your data
#     # n_channels=3 for RGB images
#     # n_classes is the number of probabilities you want to get per pixel
#     net = UNet(n_channels=1, n_classes=1, bilinear=True)
#
#     logging.info(f'Network:\n'
#                  f'\t{net.n_channels} input channels\n'
#                  f'\t{net.n_classes} output channels (classes)\n'
#                  f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
#
#     if args.load:
#         net.load_state_dict(torch.load(args.load, map_location=device))
#         logging.info(f'Model loaded from {args.load}')
#
#     net.to(device=device)
#     try:
#         train_net(net=net,
#                   epochs=args.epochs,
#                   batch_size=args.batch_size,
#                   learning_rate=args.lr,
#                   device=device,
#                   img_scale=args.scale,
#                   val_percent=args.val / 100,
#                   amp=args.amp)
#     except KeyboardInterrupt:
#         torch.save(net.state_dict(), 'INTERRUPTED.pth')
#         logging.info('Saved interrupt')
#         sys.exit(0)
