from dataloader.dataloader import SingleModalDetectionDataset
from dataloader.dataloader import MultiModalDetectionDataset
from dataloader.dataloader import ToTensor
from utils.utils import Utils
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import numpy as np
import albumentations as alb

## This is the way to bypass the transformation after you split the dataset in train and validation
## https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/2
## here we have only colorspace transformations
class DatasetTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None, modality='single'):

        self.subset = subset
        self.transform = transform
        self.modality = modality # single or multimodal
        
    def __getitem__(self, index):
        if self.modality == 'single':

            imgs, targets = self.subset[index]

            if self.transform:

                imgs = self.transform(imgs)
                
                # before_targets = targets.copy()

                # transformed_data = self.transform(image=np.array(imgs, copy=True), bboxes=targets['boxes'], labels=targets['labels'])

                # transformed_imgs = transformed_data['image']
                # transformed_bboxes = transformed_data['bboxes']
                # transformed_labels = transformed_data['labels']

                # imgs = transformed_imgs.float() / 255.0

                # if isinstance(imgs, np.ndarray):
                #     imgs = torch.from_numpy(imgs).float().unsqueeze(0)
                
                # if isinstance(transformed_bboxes, list):
                #     targets['boxes'] = torch.tensor(transformed_bboxes)
                # else:
                #     targets['boxes'] = transformed_bboxes.detach()

                # if isinstance(transformed_labels, list):
                #     targets['labels'] = torch.tensor(transformed_labels)
                # else:
                #     targets['labels'] = transformed_labels.detach()

                # if(len(targets['boxes']) == 0):
                #     targets = before_targets.copy()

                # for idx, bbox in enumerate(targets['boxes']):
                #     _, cols, rows = imgs.shape
                #     x_min, y_min, x_max, y_max = bbox
                #     rows = rows - 1
                #     cols = cols - 1
                    
                #     x_min = min(max(0, x_min), rows)
                #     y_min = min(max(0, y_min), cols)
                #     x_max = max(min(rows, x_max), 0)
                #     y_max = max(min(cols, y_max), 0)

                #     targets['boxes'][idx] = torch.Tensor([x_min, y_min, x_max, y_max]).to(torch.float64)

            return imgs, targets

        elif self.modality == 'multimodal':

            imgs_rgb, targets_rgb, imgs_ir, targets_ir = self.subset[index]

            if self.transform:
                
                imgs_rgb = T.ToPILImage()(imgs_rgb)
                imgs_ir = T.ToPILImage()(imgs_ir)

                before_targets_rgb = targets_rgb.copy()
                before_targets_ir = targets_ir.copy()

                transformed = self.transform(image=np.array(imgs_rgb, copy=True), bboxes=targets_rgb['boxes'], labels=targets_rgb['labels'],
                                                image1=np.array(imgs_ir, copy=True), bboxes1=targets_ir['boxes'], labels1=targets_ir['labels'],
                )

                transformed_rgb = transformed['image']
                transformed_ir = transformed['image1']

                transformed_bboxes_rgb = transformed['bboxes']
                transformed_bboxes_ir = transformed['bboxes1']

                transformed_labels_rgb = transformed['labels']
                transformed_labels_ir = transformed['labels1']

                imgs_rgb = transformed_rgb.float() / 255.0

                imgs_ir = transformed_ir.float() / 255.0

                if isinstance(imgs_ir, np.ndarray):
                    imgs_ir = torch.from_numpy(imgs_ir).float().unsqueeze(0)
                
                if isinstance(transformed_bboxes_rgb, list):
                    targets_rgb['boxes'] = torch.tensor(transformed_bboxes_rgb)
                else:
                    targets_rgb['boxes'] = transformed_bboxes_rgb.detach()

                if isinstance(transformed_labels_rgb, list):
                    targets_rgb['labels'] = torch.tensor(transformed_labels_rgb)
                else:
                    targets_rgb['labels'] = transformed_labels_rgb.detach()

                if isinstance(transformed_bboxes_ir, list):
                    targets_ir['boxes'] = torch.tensor(transformed_bboxes_ir)
                else:
                    targets_ir['boxes'] = transformed_bboxes_ir.detach()

                if isinstance(transformed_labels_ir, list):
                    targets_ir['labels'] = torch.tensor(transformed_labels_ir)
                else:
                    targets_ir['labels'] = transformed_labels_ir.detach()

                if(len(targets_rgb['boxes']) == 0):
                    targets_rgb = before_targets_rgb.copy()
                    targets_ir = before_targets_ir.copy()

            return imgs_rgb, targets_rgb, imgs_ir, targets_ir
        
    def __len__(self):
        return len(self.subset)


class SingleModalDataModule(pl.LightningDataModule):
    def __init__(self, dataset, path_images_train, path_images_test, batch_size=4,
                num_workers=4, ext='.png', seed=123, split_ratio_train_valid=0.8, modality='rgb',
                data_augmentation=None, fixed_transformations=None, ablation_flag=False
                ):
        super().__init__()

        self.ablation_flag = ablation_flag

        train_dataset = SingleModalDetectionDataset(
            dataset=dataset,
            path_images=path_images_train,
            modality=modality,
            transforms=None,
            ext=ext,
            train=True,
        )

        train_dataset, valid_dataset = Utils().split_dataset(train_dataset, 
                                                            split_ratio=split_ratio_train_valid, 
                                                            seed=seed)

        train_dataset = DatasetTransform(
            train_dataset, transform=data_augmentation, modality='single'
        )

        valid_dataset = DatasetTransform(
            valid_dataset, transform=fixed_transformations, modality='single' 
        )

        self.train_dataloader_modality = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=Utils().collate_fn,
                        drop_last=True,
                        num_workers=num_workers,
                        persistent_workers=True
                    )

        self.valid_dataloader_modality = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=Utils().collate_fn,
                        drop_last=True,
                        num_workers=num_workers,
                        persistent_workers=True
                    )

        self.test_dataset = SingleModalDetectionDataset(
                            dataset=dataset,
                            path_images=path_images_test,
                            modality=modality,
                            transforms=fixed_transformations,
                            ext=ext,
                            train=False,
                        )

        self.test_dataloader_modality = torch.utils.data.DataLoader(
                        self.test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=Utils().collate_fn,
                        drop_last=True,
                        num_workers=num_workers,
                        persistent_workers=True
                    )
        
        if(self.ablation_flag):
            self.valid_dataloader_modality = self.test_dataloader_modality

    def train_dataloader(self):
        return self.train_dataloader_modality

    def val_dataloader(self):
        return self.valid_dataloader_modality

    def test_dataloader(self):
        return self.test_dataloader_modality


class MultiModalDataModule(pl.LightningDataModule):
    def __init__(self, dataset, path_images_train_rgb, path_images_train_ir, path_images_test_rgb, path_images_test_ir,
                 batch_size=4, num_workers=4, ext='.png', seed=123, split_ratio_train_valid=0.8,
                data_augmentation=None, fixed_transformations=None, ablation_flag=False):
        super().__init__()

        self.ablation_flag = ablation_flag
        
        train_dataset = MultiModalDetectionDataset(
            dataset=dataset,
            path_images_rgb=path_images_train_rgb,
            path_images_ir=path_images_train_ir,
            modality="both",
            transforms_rgb=None, 
            transforms_ir=None,
            ext=ext,
            train=True,
        )

        train_dataset, valid_dataset = Utils().split_dataset(train_dataset, 
                                                            split_ratio=split_ratio_train_valid, 
                                                            seed=seed)

        train_dataset = DatasetTransform(
            train_dataset, transform=data_augmentation, modality='multimodal'
        )

        valid_dataset = DatasetTransform(
            valid_dataset, transform=fixed_transformations, modality='multimodal'
        )
        
        self.train_dataloader_multimodality = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=Utils().collate_fn,
                        drop_last=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True
                    )

        self.valid_dataloader_multimodality = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=Utils().collate_fn,
                        drop_last=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True
                    )

        self.test_dataloader_multimodality = torch.utils.data.DataLoader(
                        MultiModalDetectionDataset(
                            dataset=dataset,
                            path_images_rgb=path_images_test_rgb,
                            path_images_ir=path_images_test_ir,
                            modality="both",
                            transforms_rgb=None, 
                            transforms_ir=None,
                            ext=ext,
                            train=False,
                        ),
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=Utils().collate_fn,
                        drop_last=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True
                    )

        if(self.ablation_flag):
            self.valid_dataloader_multimodality = self.test_dataloader_multimodality
            
    def train_dataloader(self):
        return self.train_dataloader_multimodality

    def val_dataloader(self):
        ## This is just to eval model every epoch for ablation curves
        return self.valid_dataloader_multimodality

    def test_dataloader(self):
        return self.test_dataloader_multimodality
