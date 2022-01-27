import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    criterion = nn.MSELoss()
    mse_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, path = batch['image'], batch['mask'], batch['path']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            # compute the MSE score
            mse_score += criterion(mask_pred, mask_true)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return mse_score
    return mse_score / num_val_batches
