import numpy as np
import os
import pandas as pd
import skimage
import torch
from torch.utils.data import Dataset


class NIH_Dataset(Dataset):
    """NIH ChestX-ray8 dataset
    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a
    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(
        self,
        data_dir,
        phase,
        views=["PA"],
        use_pathologies=[],
        nrows=None,
        unique_patients=False,
        pathology_masks=False,
        transform=None,
    ):
        super(NIH_Dataset, self).__init__()
        self.imgdir = os.path.join(data_dir, 'images')
        self.csvpath = os.path.join(data_dir, 'Data_Entry_2017.csv')
        self.bbox_list_path = os.path.join(data_dir, 'BBox_List_2017.csv')
        self.phase = phase
        self.views = views
        self.pathology_masks = pathology_masks

        self.pathologies = [
            "No Finding", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Hernia",
            "Nodule", "Pneumothorax", "Pneumonia", "Pleural_Thickening",
        ]
        self.use_pathologies = use_pathologies if len(use_pathologies) > 0 else self.pathologies

        # Load data
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        if phase == 'train':
            with open(os.path.join(data_dir, 'train_val_list.txt')) as f:
                filelist = [x.strip() for x in f.readlines()]
            if unique_patients:
                patients = set([x.split('_')[0] for x in filelist])
                filelist = sorted([x + '_000.png' for x in patients])
                filelist = filelist[:20000]
            else:
                filelist = filelist[:70000]
        elif phase == 'val':
            with open(os.path.join(data_dir, 'train_val_list.txt')) as f:
                filelist = [x.strip() for x in f.readlines()]
            if unique_patients:
                patients = set([x.split('_')[0] for x in filelist])
                filelist = sorted([x + '_000.png' for x in patients])
                filelist = filelist[20000:]
            else:
                filelist = filelist[70000:]
        elif phase == 'test':
            with open(os.path.join(data_dir, 'test_list.txt')) as f:
                filelist = [x.strip() for x in f.readlines()]
            if unique_patients:
                patients = set([x.split('_')[0] for x in filelist])
                filelist = [x + '_000.png' for x in patients]
        
        # Remove images with view position other than specified
        self.csv = self.csv[self.csv['View Position'].isin(views)]
        # Keep images with pathologies specified
        self.csv = self.csv[self.csv['Finding Labels'].isin(self.use_pathologies)]
        # if unique_patients:
        #     self.csv = self.csv.groupby("Patient ID").first()
        self.csv = self.csv[self.csv['Image Index'].isin(filelist)]
        self.csv = self.csv.reset_index(drop=True)

        ####### pathology masks ########
        # load nih pathology masks
        self.pathology_maskscsv = pd.read_csv(self.bbox_list_path,
                                              names=["Image Index", "Finding Label", "x", "y", "w", "h", "_1", "_2", "_3"],
                                              skiprows=1)

        # change label name to match
        self.pathology_maskscsv.loc[self.pathology_maskscsv["Finding Label"] == "Infiltrate", "Finding Label"] = "Infiltration"
        self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])

        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        self.data_aug = transform

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["label"] = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgdir, imgid)
        img = skimage.io.imread(img_path, as_gray=True)
        img = img.astype(float) / 255
        img = torch.as_tensor(img, dtype=torch.float).unsqueeze(0)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        sample['img'] = self.data_aug(img)

        return sample

    def get_mask_dict(self, image_name, this_size):
        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.pathology_maskscsv[self.pathology_maskscsv["Image Index"] == image_name]
        path_mask = {}

        for i in range(len(images_with_masks)):
            row = images_with_masks.iloc[i]

            # Don't add masks for labels we don't have
            if row["Finding Label"] in self.pathologies:
                mask = np.zeros([this_size, this_size])
                xywh = np.asarray([row.x, row.y, row.w, row.h])
                xywh = xywh * scale
                xywh = xywh.astype(int)
                mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

                # Resize so image resizing works
                mask = mask[None, :, :]

                path_mask[self.pathologies.index(row["Finding Label"])] = mask
        return path_mask

