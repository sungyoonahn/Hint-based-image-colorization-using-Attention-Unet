import torch
import tqdm
import cv2
import os
import shutil
from utils import AverageMeter, ssim, save_img, psnr
from dataloader import tensor2im

# to use Tenserboard
from tensorboardX import SummaryWriter

##TRAIN##
def train(train_loader, model, criterion, optimizer, epoch):
    print('Starting training epoch {}'.format(epoch))
    model.train()
    use_cuda = True
    # Prepare value counters and timers
    losses = AverageMeter()

    # Tenserboard writer
    writer = SummaryWriter("logs")

    for i, data in enumerate(tqdm.tqdm(train_loader)):
        if use_cuda:
            l = data["l"].to('cuda')
            ab = data["ab"].to('cuda')
            hint = data["hint"].to('cuda')

        gt_image = torch.cat((l, ab), dim=1)
        hint_image = torch.cat((l, hint), dim=1)

        # Run forward pass
        output_hint = model(hint_image)
        loss = criterion(output_hint, gt_image)
        losses.update(loss.item(), hint_image.size(0))
        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses))

    print('Finished training epoch {}'.format(epoch))


def validate(val_loader, model, criterion, save_images, epoch):
    model.eval()
    use_cuda = True
    # Prepare value counters and timers
    losses = AverageMeter()

    # Sum of ssim, psnr
    sum_ssim = 0
    sum_psnr = 0

    shutil.rmtree("outputs/Output")
    os.makedirs('outputs/Output', exist_ok=True)

    for i, data in enumerate(tqdm.tqdm(val_loader)):
        if use_cuda:
            l = data["l"].to('cuda')
            ab = data["ab"].to('cuda')
            hint = data["hint"].to('cuda')

        gt_image = torch.cat((l, ab), dim=1)
        hint_image = torch.cat((l, hint), dim=1)
        output_hint = model(hint_image)

        loss = criterion(output_hint, gt_image)
        losses.update(loss.item(), hint_image.size(0))

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 100 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), loss=losses))
        out_hint_np = tensor2im(output_hint)
        out_hint_bgr = cv2.cvtColor(out_hint_np, cv2.COLOR_LAB2BGR)

        hint_np = tensor2im(hint_image)
        hint_bgr = cv2.cvtColor(hint_np, cv2.COLOR_LAB2BGR)

        gt_np = tensor2im(gt_image)
        gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_LAB2BGR)

        save_img(gt_bgr, hint_bgr, out_hint_bgr, i)

        # get ssim, psnr values
        sum_ssim += SSIM(gt_bgr, hint_bgr)
        sum_psnr += psnr(gt_bgr, hint_bgr)

    # save mean of SSIM, PSNR using Tensorboard writer
    writer.add_scalar('SSIM on validation data', sum_ssim/len(val_loader.dataset), epoch+1)
    writer.add_scalar('PSNR on validation data', sum_psnr/len(val_loader.dataset), epoch+1)

    # print mean of SSIM, PSNR
    print('[epoch %d] SSIM: avg = {}'.format(epoch+1, sum_ssim/len(val_loader.dataset)))
    print('[epoch %d] PSNR: avg = {}'.format(epoch+1, sum_psnr/len(val_loader.dataset)))

    print('Finished validation.')
    return losses.avg, sum_ssim, sum_psnr


def test(test_loader, model):
    model.eval()
    use_cuda= True

    for i, data in enumerate(tqdm.tqdm(test_loader)):
        if use_cuda:
            l = data["l"].to('cuda')
            hint = data["hint"].to('cuda')
            file_name = data["file_name"][0]

        hint_image = torch.cat((l, hint), dim=1)
        output_hint = model(hint_image)

        out_hint_np = tensor2im(output_hint)
        out_hint_bgr = cv2.cvtColor(out_hint_np, cv2.COLOR_LAB2BGR)
        cv2.imwrite("outputs/test/"+file_name, out_hint_bgr)

    print('Finished test.')
