import os
import torch
import torch.utils.data
from torch import nn

from dataloader import ColorHintDataset, tensor2im
from train import train, validate
# from myUnet import Unet
from mynet2 import Unet


def main():
    ## DATALOADER ##
    # Change to your data root directory
    root_path = "data/"
    # Depend on runtime setting
    use_cuda = True

    train_dataset = ColorHintDataset(root_path, 128)
    train_dataset.set_mode("training")

    test_dataset = ColorHintDataset(root_path, 128)
    test_dataset.set_mode("validation")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    model = Unet()
    # print(model)
    PATH = "model-epoch-8-losses-0.00763.pth"
    model.load_state_dict(torch.load(PATH))

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

    # Move model and loss function to GPU
    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    # Make folders and set parameters
    os.makedirs('outputs/GroundTruth', exist_ok=True)
    os.makedirs('outputs/Hint', exist_ok=True)
    os.makedirs('outputs/Output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    save_images = True
    best_losses = 1e10
    epochs = 250
    # Train model
    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(train_dataloader, model, criterion, optimizer, epoch)
        with torch.no_grad():
            losses = validate(test_dataloader, model, criterion, save_images, epoch)
        # Save checkpoint and replace old best model if current model is better
        if losses < best_losses:
            best_losses = losses
            torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.5f}.pth'.format(epoch + 1, losses))


if __name__ == '__main__':
    main()
