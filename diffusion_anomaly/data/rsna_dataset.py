import os
import pandas as pd
import pydicom
import skimage.io
import numpy as np
import torch
from typing import Union

from .base_dataset import Dataset


def normalize(img: np.ndarray, maxval: Union[float, int]):
    if img.max() > maxval:
        raise Exception('max image value higher than expected bound.')
    
    img = img.astype(np.float) / maxval
    return img


class RSNA_Pneumonia_Dataset(Dataset):
    """RSNA Pneumonia Detection Challenge
    Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert
    Annotations of Possible Pneumonia.
    Shih, George, Wu, Carol C., Halabi, Safwan S., Kohli, Marc D., Prevedello, Luciano M.,
    Cook, Tessa S., Sharma, Arjun, Amorosa, Judith K., Arteaga, Veronica, Galperin-Aizenberg,
    Maya, Gill, Ritu R., Godoy, Myrna C.B., Hobbs, Stephen, Jeudy, Jean, Laroia, Archana,
    Shah, Palmi N., Vummidi, Dharshan, Yaddanapudi, Kavitha, and Stein, Anouk.
    Radiology: Artificial Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.
    More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
    Challenge site:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    JPG files stored here:
    https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
    """

    def __init__(self,
                 imgdir,
                 has_label,
                 csvpath="kaggle_stage_2_train_labels.csv.zip",
                 dicomcsvpath="kaggle_stage_2_train_images_dicom_headers.csv.gz",
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True,
                 pathology_masks=False,
                 extension=".jpg",
                 normal_only=False
                 ):

        super(RSNA_Pneumonia_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgdir = imgdir
        self.has_label = has_label
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Pneumonia", "Lung Opacity"]
        self.pathologies = sorted(self.pathologies)
        self.classes = ['normal', 'pneumonia']

        self.extension = extension
        self.use_pydicom = (extension == ".dcm")

        if has_label:
            # Load data
            self.csvpath = csvpath
            self.raw_csv = pd.read_csv(self.csvpath, nrows=nrows)

            # The labels have multiple instances for each mask
            # So we just need one to get the target label
            self.csv = self.raw_csv.groupby("patientId").first()

            # self.dicomcsvpath = dicomcsvpath
            # self.dicomcsv = pd.read_csv(self.dicomcsvpath, nrows=nrows, index_col="PatientID")

            # self.csv = self.csv.join(self.dicomcsv, on="patientId")

            # Remove images with view position other than specified
            # self.csv["view"] = self.csv['ViewPosition']
            # self.limit_to_selected_views(views)

            # Only return normal images
            if normal_only:
                self.csv = self.csv[self.csv['Target'] == 0]

            self.csv = self.csv.reset_index()

            # Get our classes.
            self.labels = []
            self.labels.append(self.csv["Target"].values)
            self.labels.append(self.csv["Target"].values)  # same labels for both

            # set if we have masks
            self.csv["has_masks"] = ~np.isnan(self.csv["x"])

            self.labels = np.asarray(self.labels).T
            self.labels = self.labels.astype(np.float32)

            # add consistent csv values

            # offset_day_int
            # TODO: merge with NIH metadata to get dates for images

            # patientid
            # self.csv["patientid"] = self.csv["patientId"].astype(str)
        
        else:
            self.img_names = sorted(os.listdir(imgdir))

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        if self.has_label:
            return len(self.labels)
        else:
            return len(self.img_names)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        if self.has_label:
            sample["label"] = self.labels[idx]
            imgid = self.csv['patientId'].iloc[idx]
            img_path = os.path.join(self.imgdir, imgid + self.extension)
        else:
            img_path = os.path.join(self.imgdir, self.img_names[idx])

        if self.use_pydicom:
            img = pydicom.filereader.dcmread(img_path).pixel_array
        else:
            img = skimage.io.imread(img_path)

        img = normalize(img, maxval=255)[None]
        img = torch.as_tensor(img, dtype=torch.float)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        # sample = apply_transforms(sample, self.transform)
        # sample = apply_transforms(sample, self.data_aug)
        sample['img'] = self.transform(img)

        return sample

    def get_mask_dict(self, image_name, this_size):

        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.raw_csv[self.raw_csv["patientId"] == image_name]
        path_mask = {}

        # All masks are for both pathologies
        for patho in ["Pneumonia", "Lung Opacity"]:
            mask = np.zeros([this_size, this_size])

            # Don't add masks for labels we don't have
            if patho in self.pathologies:

                for i in range(len(images_with_masks)):
                    row = images_with_masks.iloc[i]
                    xywh = np.asarray([row.x, row.y, row.width, row.height])
                    xywh = xywh * scale
                    xywh = xywh.astype(int)
                    mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

            # Resize so image resizing works
            mask = mask[None, :, :]

            path_mask[self.pathologies.index(patho)] = mask
        return path_mask