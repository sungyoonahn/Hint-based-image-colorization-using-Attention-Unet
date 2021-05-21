import cv2
from skimage.metrics import structural_similarity
from pytorch_msssim import ssim
import pytorch_ssim
import torch
from torchvision.transforms import ToTensor
from torch.autograd import Variable


## calculate loss per image##
class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial'''
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

##SSIM##
def SSIM(img_A, img_B):
    img1 = ToTensor()(img_A).unsqueeze(0)
    img2 = ToTensor()(img_B).unsqueeze(0)
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    ssim_val = round(ssim(img1, img2).item(),2)
    return ssim_val

def psnr(img_A, img_B):
    score = cv2.PSNR(img_A, img_B)
    return score

##SAVE IMG##
def save_img(gt, hint, output, num):
    SSIM_VAL = SSIM(gt, output)
    PSNR = psnr(gt, output)
    cv2.imwrite("outputs/GroundTruth/"+str(num)+"gt.png", gt)
    cv2.imwrite("outputs/Hint/"+str(num)+"hint.png", hint)
    cv2.imwrite("outputs/Output/"+str(num)+"_ssim:"+str(SSIM_VAL)+"_psnr:"+str(PSNR)+".png", output)



