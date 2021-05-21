import os
import torch
import torch.utils.data
from torch import nn

from dataloader import ColorHintDataset, tensor2im
from model import ColorizationNet
from fastai_Unet import build_res_unet
from train import train, validate, test
from utils import ssim
from myUnet import Unet


def main():
    ## DATALOADER ##
    # Change to your data root directory
    root_path = "data/"
    # Depend on runtime setting
    use_cuda = True

    test_dataset = ColorHintDataset(root_path, 128)
    test_dataset.set_mode("validation")

    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    model = Unet()
    PATH = "model-epoch-6-losses-0.020.pth"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    os.makedirs('outputs/test', exist_ok=True)

    # Move model and loss function to GPU
    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    # Make folders and set parameters
    epoch = 1
    save_images = True
    with torch.no_grad():
        losses = validate(test_dataloader, model, criterion, save_images, epoch)


if __name__ == '__main__':
    main()
