import os
import torch
import torch.utils.data
from torch import nn

from dataloader import ColorHintDataset, tensor2im
from train import train, validate, test
# from myUnet import Unet
from mynet2 import Unet

import matplotlib.pyplot as plt
import numpy as np
import pylab

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


    ## calculate and save psnr, ssim ##

    # change to your Output data directory
    output_path = "/outputs/Output"
    file_list = os.listdir(output_path)
        
    # change to your epoch value
    epoch = 250
    # change to your validation dataset length
    val_dataset_len = 500

    ssim = np.zeros(epoch)
    psnr = np.zeros(epoch)

    for img_name in file_list:
        # print(img_name)
        name = img_name.replace('.png', '')   # remove '.png'
        temp = name.split('_')
        ssim[int(temp[0])-1] += float(temp[3].replace('ssim:', ''))
        psnr[int(temp[0])-1] += float(temp[4].replace('psnr:', ''))

    ssim_avg = ssim/val_dataset_len
    psnr_avg = psnr/val_dataset_len

    print(ssim_avg)
    print(psnr_avg)

    np.save(os.path.join('./', 'ssim_avg.npy'), ssim_avg)
    np.save(os.path.join('./', 'psnr_avg.npy'), psnr_avg)

    # plot and save ssim curve
    plt.figure()
    plt.title('ssim')
    pylab.xlim(0, epoch + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, epoch + 1), ssim_avg, label='ssim_avg')
    plt.legend()
    plt.savefig(os.path.join('./', 'ssim.pdf'))
    plt.show()
    plt.close()

    # plot and save psnr curve
    plt.figure()
    plt.title('pnsr')
    pylab.xlim(0, epoch + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, epoch + 1), psnr_avg, label='psnr_avg')
    plt.legend()
    plt.savefig(os.path.join('./', 'psnr.pdf'))
    plt.show()
    plt.close()

    # plot and save ssim, psnr curve
    plt.figure()
    plt.title('ssim & pnsr')
    pylab.xlim(0, epoch + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, epoch + 1), ssim_avg, label='ssim_avg')
    plt.plot(range(1, epoch + 1), psnr_avg, label='psnr_avg')
    plt.legend()
    plt.savefig(os.path.join('./', 'ssim_psnr.pdf'))
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
