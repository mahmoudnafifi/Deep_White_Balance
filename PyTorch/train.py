"""
 Training
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from arch import deep_wb_model
import arch.splitNetworks as splitter

try:
    from torch.utils.tensorboard import SummaryWriter

    use_tb = True
except ImportError:
    use_tb = False

from utilities.dataset import BasicDataset
from utilities.loss_func import mae_loss

from torch.utils.data import DataLoader, random_split


def train_net(net,
              device,
              epochs=110,
              batch_size=32,
              lr=0.0001,
              val_percent=0.1,
              lrdf=0.5,
              lrdp=25,
              fold=0,
              chkpointperiod=1,
              trimages=12000,
              patchsz=128,
              patchnum=4,
              validationFrequency=4,
              dir_img='../dataset',
              save_cp=True):
    dir_checkpoint = f'checkpoints_{fold}/'
    dataset = BasicDataset(dir_img, fold=fold, patch_size=patchsz, patch_num_per_image=patchnum, max_trdata=trimages)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    if use_tb:
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Patches/image:   {patchnum}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        TensorBoard:     {use_tb}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs_ = batch['image']
                awb_gt_ = batch['gt-AWB']
                t_gt_ = batch['gt-T']
                s_gt_ = batch['gt-S']
                assert imgs_.shape[1] == net.n_channels * patchnum, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert awb_gt_.shape[1] == net.n_channels * patchnum, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded AWB GT images have {awb_gt_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert t_gt_.shape[1] == net.n_channels * patchnum, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded Tungsten WB GT images have {t_gt_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert s_gt_.shape[1] == net.n_channels * patchnum, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded Shade WB GT images have {s_gt_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

                for j in range(patchnum):
                    imgs = imgs_[:, (j * 3): 3 + (j * 3), :, :]
                    awb_gt = awb_gt_[:, (j * 3): 3 + (j * 3), :, :]
                    t_gt = t_gt_[:, (j * 3): 3 + (j * 3), :, :]
                    s_gt = s_gt_[:, (j * 3): 3 + (j * 3), :, :]

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    awb_gt = awb_gt.to(device=device, dtype=torch.float32)
                    t_gt = t_gt.to(device=device, dtype=torch.float32)
                    s_gt = s_gt.to(device=device, dtype=torch.float32)

                    imgs_pred = net(imgs)
                    loss = mae_loss.compute(imgs_pred, torch.cat((awb_gt, t_gt, s_gt), dim=1))
                    epoch_loss += loss.item()
                    if use_tb:
                        writer.add_scalar('Loss/train', loss.item(), global_step)

                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.update(np.ceil(imgs.shape[0] / patchnum))
                    global_step += 1

        if (epoch + 1) % validationFrequency == 0:
            if use_tb:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_score = vald_net(net, val_loader, device)
            logging.info('Validation MAE: {}'.format(val_score))
            if use_tb:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Loss/test', val_score, global_step)
                writer.add_images('images', imgs, global_step)
                writer.add_images('result-awb', imgs_pred[:, :3, :, :], global_step)
                writer.add_images('result-t', imgs_pred[:, 3:6, :, :], global_step)
                writer.add_images('result-s', imgs_pred[:, 6:, :, :], global_step)
                writer.add_images('GT_awb', awb_gt, global_step)
                writer.add_images('GT-t', t_gt, global_step)
                writer.add_images('GT-s', s_gt, global_step)

        scheduler.step()

        if save_cp and (epoch + 1) % chkpointperiod == 0:
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')

            torch.save(net.state_dict(), dir_checkpoint + f'deep_WB_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')

    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')
    torch.save(net.state_dict(), 'models/' + 'net.pth')
    logging.info('Saved trained model!')
    logging.info('Saving each auto-encoder model separately')
    net_awb, net_t, net_s = splitter.splitNetworks(net)
    torch.save(net_awb.state_dict(), 'models/' + 'net_awb.pth')
    torch.save(net_t.state_dict(), 'models/' + 'net_t.pth')
    torch.save(net_s.state_dict(), 'models/' + 'net_s.pth')
    logging.info('Saved trained models!')
    if use_tb:
        writer.close()
    logging.info('End of training')


def vald_net(net, loader, device):
    """Evaluation using MAE"""
    net.eval()
    n_val = len(loader) + 1
    mae = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs_ = batch['image']
            awb_gt_ = batch['gt-AWB']
            t_gt_ = batch['gt-T']
            s_gt_ = batch['gt-S']
            patchnum = imgs_.shape[1] / 3
            assert imgs_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'

            assert awb_gt_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded AWB GT images have {awb_gt_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'

            assert t_gt_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded Tungsten WB GT images have {t_gt_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'

            assert s_gt_.shape[1] == net.n_channels * patchnum, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded Shade WB GT images have {s_gt_.shape[1] / patchnum} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs_[:, 0:3, :, :]
            awb_gt = awb_gt_[:, 0:3, :, :]
            t_gt = t_gt_[:, 0:3, :, :]
            s_gt = s_gt_[:, 0:3, :, :]
            imgs = imgs.to(device=device, dtype=torch.float32)
            awb_gt = awb_gt.to(device=device, dtype=torch.float32)
            t_gt = t_gt.to(device=device, dtype=torch.float32)
            s_gt = s_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                imgs_pred = net(imgs)
                loss = mae_loss.compute(imgs_pred, torch.cat((awb_gt, t_gt, s_gt), dim=1))
                mae = mae + loss

            pbar.update(np.ceil(imgs.shape[0] / patchnum))

    net.train()
    return mae / n_val


def get_args():
    parser = argparse.ArgumentParser(description='Train deep WB editing network.')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=110,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-l', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-vf', '--validation-frequency', dest='val_frq', type=int, default=5,
                        help='Validation frequency.')
    parser.add_argument('-d', '--fold', dest='fold', type=int, default=1,
                        help='Testing fold to be excluded. Use --fold 0 to use all Set1 training data')
    parser.add_argument('-p', '--patches-per-image', dest='patchnum', type=int, default=4,
                        help='Number of training patches per image')
    parser.add_argument('-s', '--patch-size', dest='patchsz', type=int, default=128,
                        help='Size of training patch')
    parser.add_argument('-t', '--num_training_images', dest='trimages', type=int, default=13333,
                        help='Number of training images. Use --num_training_images 0 to use all training images')
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod', type=int, default=5,
                        help='Number of epochs to save a checkpoint')
    parser.add_argument('-ldf', '--learning-rate-drop-factor', dest='lrdf', type=float, default=0.5,
                        help='Learning rate drop factor')
    parser.add_argument('-ldp', '--learning-rate-drop-period', dest='lrdp', type=int, default=25,
                        help='Learning rate drop period')
    parser.add_argument('-trd', '--training_dir', dest='trdir', default='../dataset/',
                        help='Training image directory')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of Deep White-Balance Editing')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = deep_wb_model.deepWBNet()
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  lrdf=args.lrdf,
                  lrdp=args.lrdp,
                  device=device,
                  fold=args.fold,
                  chkpointperiod=args.chkpointperiod,
                  trimages=args.trimages,
                  val_percent=args.val / 100,
                  validationFrequency=args.val_frq,
                  patchsz=args.patchsz,
                  patchnum=args.patchnum,
                  dir_img=args.trdir
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'bkupCheckPoint.pth')
        logging.info('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
