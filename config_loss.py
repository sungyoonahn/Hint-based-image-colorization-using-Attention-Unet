import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torchvision.models as models
# from google.colab import drive
import os

# drive.mount("/content/gdrive")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/gdrive/MyDrive/deep

from dataload import CustomImageDataset
from train import train, eval
from utils import plot

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
    epoch = 10
    batch_size = 64
    learning_rate = 1e-3

    # Data Transform
    transforms_train = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.ToTensor()])

    # Load Data
    train_data_set = CustomImageDataset(data_set_path=data_path + "/train", transforms=transforms_train)
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

    test_data_set = CustomImageDataset(data_set_path=data_path + "/val", transforms=transforms_test)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

    if not (train_data_set.num_classes == test_data_set.num_classes):
        print("error: Numbers of class in training set and test set are not equal")
        exit()

    # optimizer, loss function
    criterion = [nn.L1Loss(),
                 nn.MSELoss(),
                 nn.CrossEntropyLoss(),
                 nn.CTCLoss(),
                 nn.NLLLoss(),
                 nn.PoissonNLLLoss(),
                 nn.GaussianNLLLoss(),
                 nn.KLDivLoss(),
                 nn.BCELoss(),
                 nn.BCEWithLogitsLoss(),
                 nn.MarginRankingLoss(),
                 nn.HingeEmbeddingLoss(),
                 nn.MultiLabelMarginLoss(),
                 nn.SmoothL1Loss(),
                 nn.SoftMarginLoss(),
                 nn.MultiLabelSoftMarginLoss(),
                 nn.CosineEmbeddingLoss(),
                 nn.MultiMarginLoss(),
                 nn.TripletMarginLoss(),
                 nn.TripletMarginWithDistanceLoss()]

    criterion_name = ["L1Loss",
                      "MSELoss",
                      "CrossEntropyLoss",
                      "CTCLoss",
                      "NLLLoss",
                      "PoissonNLLLoss",
                      "GaussianNLLLoss",
                      "KLDivLoss",
                      "BCELoss",
                      "BCEWithLogitsLoss",
                      "MarginRankingLoss",
                      "HingeEmbeddingLoss",
                      "MultiLabelMarginLoss",
                      "SmoothL1Loss",
                      "SoftMarginLoss",
                      "MultiLabelSoftMarginLoss",
                      "CosineEmbeddingLoss",
                      "MultiMarginLoss",
                      "TripletMarginLoss",
                      "TripletMarginWithDistanceLoss"]

    success_loss = {}

    for each_loss in criterion:

        # Model (resnet18)
        num_classes = train_data_set.num_classes
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Linear(512, num_classes)

        # Training Model
        train_loss, train_accuracy = [], []
        val_loss, val_accuracy = [], []

        # Load Model
        model = resnet.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0005)
        os.chdir(save_path)

        try:
            for e in range(epoch):
                # Train, Val
                train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, each_loss, e)
                val_epoch_loss, val_epoch_acc = eval(model, test_loader, each_loss)
                train_loss.append(train_epoch_loss)
                train_accuracy.append(train_epoch_acc)
                val_loss.append(val_epoch_loss)
                val_accuracy.append(val_epoch_acc)

                try:
                    os.mkdir(criterion_name[criterion.index(each_loss)])
                except:
                    pass

                # Plot Acc and Loss for Train and Val Results
                plot(train_accuracy, val_accuracy, train_loss, val_loss,
                     save_path + "/" + criterion_name[criterion.index(each_loss)])

                # Saves Each Epoch Results (EPOCH IS SAVED IN .pth NAME)
                torch.save(model.state_dict(), save_path + '/' + criterion_name[criterion.index(each_loss)] +
                           "/" + criterion_name[criterion.index(each_loss)] + ' model_epoch_{}.pth'.format(e))

                success_loss[criterion_name[criterion.index(each_loss)]] = sum(val_accuracy) / len(val_accuracy)

        except RuntimeError as e:
            txt_file = open(criterion_name[criterion.index(each_loss)] + ".txt", 'w')
            print(e, file=txt_file)
            txt_file.close()
            print(each_loss, end="    ")
            print("Can't")

        except TypeError as e:
            txt_file = open(criterion_name[criterion.index(each_loss)] + ".txt", 'w')
            print(e, file=txt_file)
            txt_file.close()
            print(each_loss, end="    ")
            print("Can't")

        except ValueError as e:
            txt_file = open(criterion_name[criterion.index(each_loss)] + ".txt", 'w')
            print(e, file=txt_file)
            txt_file.close()
            print(each_loss, end="    ")
            print("Can't")

        print()
    print("success loss function (matched mean val_acc) : ", end='')
    print(success_loss)

print("success loss function (matched mean val_acc) : ", end='')
print(success_loss)