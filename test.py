import os
import torch
import torch.utils.data
from torch import nn

from dataloader import ColorHintDataset, tensor2im
from model import ColorizationNet
from fastai_Unet import build_res_unet
from train import train, validate, test


def main():
    ## DATALOADER ##
    # Change to your data root directory
    root_path = "test_data/"
    # Depend on runtime setting
    use_cuda = True

    test_dataset = ColorHintDataset(root_path, 128)
    test_dataset.set_mode("testing")

    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    model = build_res_unet()
    print(model)
    PATH = "model-epoch-17-losses-0.015.pth"
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
    with torch.no_grad():
        test(test_dataloader, model, criterion)

if __name__ == '__main__':
    main()
