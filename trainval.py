import argparse
import numpy as np
import os
from PIL import Image
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

import model
import data
import utils
import meter


def main():
    parser = argparse.ArgumentParser(description='Set config and checkpoint.')
    parser.add_argument('--config', metavar='str', required=True, type=str)
    parser.add_argument('--checkpoint', metavar='file', type=str, default='')
    args = parser.parse_args()
    config = utils.load_config(args.config)

    os.makedirs(os.path.join('checkpoints', config.exp_name), exist_ok=True)
    # save config file to checkpoint dir
    shutil.copy(
        os.path.join('configs', args.config + '.yaml'),
        os.path.join('checkpoints', config.exp_name, 'config.yaml')
    )
    writer = SummaryWriter(log_dir=os.path.join('checkpoints', config.exp_name))

    train_loader = data.build_dataloader(config, phase='train_normal')
    # val_loader = data.build_dataloader(config, phase='test')
    train_iter = utils.cycle(train_loader)
    
    trainer = model.get_model(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        args.start_step = trainer.step + 1
    else:
        args.start_step = 1
    
    best_metrics = float('inf')
    train_meters = meter.AverageMeterDict()
    for step in range(args.start_step, config.num_steps + 1):
        feeddict = next(train_iter)
        feeddict['img'] = feeddict['img'].cuda()

        loss_dict = trainer.train_step(feeddict)
        train_meters.update(loss_dict, n=feeddict['img'].size(0))

        trainer.adjust_learning_rate()

        if step % config.print_freq == 0:
            for key in train_meters.keys():
                writer.add_scalar(f'TrainLoss/{key}', train_meters[key].avg, step)
                print(f'step {step}, train {key} {train_meters[key].avg}')
            train_meters.reset()
        
        if step % config.val_freq == 0:
            img = feeddict['img'].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            for i in range(img.shape[0]):
                tmp = Image.fromarray(img[i, 0])
                tmp.save(os.path.join('checkpoints', config.exp_name, f'train_sample_{i}.jpg'))
            
            sampled_imgs = trainer.sample(batch_size=24)
            sampled_imgs = sampled_imgs.cpu().numpy()
            sampled_imgs = (sampled_imgs * 255).astype(np.uint8)
            for i in range(24):
                tmp = Image.fromarray(sampled_imgs[i, 0])
                tmp.save(os.path.join('checkpoints', config.exp_name, f'val_sample_{i}.jpg'))

            # val_meters = val_epoch(val_loader, trainer, epoch)
            # for key in val_meters.keys():
            #     writer.add_scalar(f'ValLoss/{key}', val_meters[key].avg, epoch)
            # if val_meters['l1_loss'].avg < best_metrics:
            #     best_metrics = val_meters['l1_loss'].avg
            #     save_checkpoint(config.exp_name, model, optimizer, epoch)
        
        if step % config.checkpoint_freq == 0:
            filename = os.path.join('checkpoints', config.exp_name, f'step{step}.pth')
            trainer.save_checkpoint(filename)


def val_epoch(val_loader, trainer, epoch):
    meters = meter.AverageMeterDict()

    for it, feeddict in enumerate(tqdm(val_loader, desc=f'Epoch {epoch}: ')):
        for key in feeddict:
            if isinstance(feeddict[key], torch.Tensor):
                feeddict[key] = feeddict[key].cuda()
        with torch.no_grad():
            for mask in feeddict['masks']:
                mask = mask.cuda()
                mask = mask.type(torch.float)
                feeddict['mask'] = mask
                output, loss_dict = trainer.val_step(feeddict)
                mask = mask.cpu()
                meters.update(loss_dict, n=feeddict['image'].size(0))
    return meters


if __name__ == '__main__':
    main()