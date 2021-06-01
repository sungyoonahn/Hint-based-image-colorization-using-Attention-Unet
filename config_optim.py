import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torchvision.models as models

from dataload import CustomImageDataset
from train import train, eval
from utils import plot
from config_settting import test_optim

# For Colab(Gdrive mount)
# from google.colab import drive
#
# drive.mount("/content/gdrive")

if __name__ == "__main__":
    # Load device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Paths
    # Data Path
    data_path = "config_data"
    # Output Path
    save_path = "outputs"

    # Hyper Parameters
    epoch =3
    batch_size = 2
    learning_rate = 1e-3

    # Data Transform
    transforms_train = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.ToTensor()])

    # Load Data
    train_data_set = CustomImageDataset(data_set_path=data_path+"/train", transforms=transforms_train)
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

    test_data_set = CustomImageDataset(data_set_path=data_path+"/val", transforms=transforms_test)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

    if not (train_data_set.num_classes == test_data_set.num_classes):
        print("error: Numbers of class in training set and test set are not equal")
        exit()

    # Model (resnet18)
    num_classes = train_data_set.num_classes
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(512, num_classes)
    # Load Model

    # optimizer, loss function

    criterion = nn.CrossEntropyLoss()

    # Training Model


    for i in range(4):
        model = resnet.to(device)
        optimizer, name = test_optim(i, model, learning_rate)
        train_loss, train_accuracy = [], []
        val_loss, val_accuracy = [], []
        for e in range(epoch):
            # Train, Val
            train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, e)
            val_epoch_loss, val_epoch_acc = eval(model, test_loader, criterion)

            train_loss.append(train_epoch_loss)
            train_accuracy.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_accuracy.append(val_epoch_acc)

            # Plot Acc and Loss for Train and Val Results
            plot(train_accuracy, val_accuracy, train_loss, val_loss, save_path, name)

            # Saves Each Epoch Results (EPOCH IS SAVED IN .pth NAME)
            torch.save(model.state_dict(), save_path+'/'+name+'_epoch_{}.pth'.format(e))
