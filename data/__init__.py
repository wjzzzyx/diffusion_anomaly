import torch
from torchvision import transforms

import data.rsna_dataset as rsna
import data.nih_dataset as nih


def build_dataloader(config, phase):
    transform = transforms.Compose([
        # xrv.datasets.XRayCenterCrop(),
        # xrv.datasets.XRayResizer(128)
        # transforms.CenterCrop(1024),
        transforms.Resize(config.img_size)
    ])

    if config.dataset == 'rsna':
        if phase == 'train':
            dataset = rsna.RSNA_Pneumonia_Dataset(
                imgdir='/mnt/sdf/yixiao/xray_data/rsna-pneumonia-detection/stage_2_train_images',
                has_label=True,
                csvpath='/mnt/sdf/yixiao/xray_data/rsna-pneumonia-detection/stage_2_train_labels.csv',
                transform=transform,
                extension='.dcm',
                normal_only=config.normal_only
            )
        elif phase == 'test':
            dataset = rsna.RSNA_Pneumonia_Dataset(
                imgdir='/mnt/sdf/yixiao/xray_data/rsna-pneumonia-detection/stage_2_test_images',
                has_label=False,
                transform=transform,
                extension='.dcm'
            )
    elif config.dataset == 'nih':
        if phase in ['train', 'val']:
            dataset = nih.NIH_Dataset(
                data_dir='/mnt/sdf/yixiao/xray_data/nih',
                phase=phase,
                views=['PA'],
                use_pathologies=['No Finding'],
                transform=transform,
            )
        elif phase == 'test':
            dataset = nih.NIH_Dataset(
                data_dir='/mnt/sdf/yixiao/xray_data/nih',
                phase=phase,
                views=['PA'],
                use_pathologies=[],
                transform=transform,
            )
    else:
        raise NotImplementedError()
    
    if phase in ['train']:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    return dataloader
        