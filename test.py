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

    ssim = np.zeros(len(file_list))
    psnr = np.zeros(len(file_list))

    for i, img_name in enumerate(file_list):
        # print(img_name)
        name = img_name.replace('.png', '')   # remove '.png'
        temp = name.split('_')
        ssim[i] += float(temp[1].replace('ssim:', ''))
        psnr[i] += float(temp[2].replace('psnr:', ''))

    ssim_avg = sum(ssim)/len(ssim)
    psnr_avg = sum(psnr)/len(psnr)

    print('Average of ssim: {}'.format(ssim_avg))
    print('Average of psnr: {}'.format(psnr_avg))

    np.save(os.path.join('./', 'ssim.npy'), ssim)
    np.save(os.path.join('./', 'psnr.npy'), psnr)

    # plot and save ssim curve
    plt.figure()
    plt.title('ssim')
    pylab.xlim(0, len(file_list) + 1)
    pylab.ylim(0, 1.1)
    plt.plot(range(1, len(file_list) + 1), ssim, label='ssim')
    plt.legend()
    plt.savefig(os.path.join('./', 'ssim.pdf'))
    plt.show()
    plt.close()

    # plot and save psnr curve
    plt.figure()
    plt.title('pnsr')
    pylab.xlim(0, len(file_list) + 1)
    pylab.ylim(0, 100)
    plt.plot(range(1, len(file_list) + 1), psnr, label='psnr')
    plt.legend()
    plt.savefig(os.path.join('./', 'psnr.pdf'))
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
