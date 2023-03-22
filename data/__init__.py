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
        if phase == 'train':
            dataset = nih.NIH_Dataset(
                imgdir='/mnt/sdi/yixiao/xray_data/nih/images',
                case_list='train_val_list.txt',
                csvpath='/mnt/sdi/yixiao/xray_data/nih/Data_Entry_2017.csv',
                bbox_list_path='/mnt/sdi/yixiao/xray_data/nih/BBox_List_2017.csv',
                views=['PA'],
                transform=transform,
            )
        elif phase == 'test':
            dataset = nih.NIH_Dataset(
                imgdir='/mnt/sdi/yixiao/xray_data/nih/images',
                case_list='test_list.txt',
                csvpath='/mnt/sdi/yixiao/xray_data/nih/Data_Entry_2017.csv',
                bbox_list_path='/mnt/sdi/yixiao/xray_data/nih/BBox_List_2017.csv',
                views=['PA'],
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
        