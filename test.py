import os
import torch
import torch.utils.data
from torch import nn

from dataloader import ColorHintDataset, tensor2im
from train import train, validate, test
# from myUnet import Unet
from mynet2 import Unet

def main():
    ## DATALOADER ##
    # Change to your data root directory
    root_path = "test_data/"
    # Depend on runtime setting
    use_cuda = True

    test_dataset = ColorHintDataset(root_path, 128)
    test_dataset.set_mode("testing")

    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    model = Unet()
    print(model)
    PATH = "model-epoch-70-losses-0.00709.pth"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    os.makedirs('outputs/test', exist_ok=True)

    # Move model and loss function to GPU
    if use_cuda:
        model = model.cuda()
    # Make folders and set parameters
    with torch.no_grad():
        test(test_dataloader, model)

if __name__ == '__main__':
    main()
