import numpy as np
from torch.utils.data import Dataset
import cv2
import os
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


# Load dataset
def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    return mask

class DefectDataset(Dataset):
    '''
    An Extensible Multi-Modality Dataset Class using albumentations
    '''
    def __init__(self, root_dir, num_classes, image_set, input_modalities = ['rgb', 'depth', 'normal', 'curvature'], transforms=None):
        self.n_class = num_classes
        self.labels_avail = True
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_set = image_set
        self.input_modalities = input_modalities
        if self.labels_avail:
            self.input_modalities.append('labels')
        self.data_filenames = dict((key, []) for key in self.input_modalities)
        self.reference_filename = os.path.join(root_dir, '{:s}.txt'.format(image_set))
        img_list = self.read_image_list(self.reference_filename)
        for img_name in img_list:
            for modality in self.input_modalities:
                if os.path.isfile(os.path.join(root_dir, modality + '/{:s}'.format(img_name))):
                    self.data_filenames[modality].append(os.path.join(root_dir, modality + '/{:s}'.format(img_name)))

    def __len__(self):
        return len(self.data_filenames[self.input_modalities[0]])

    def read_image_list(self, filename):
        list_file = open(filename, 'r')
        img_list = []
        while True:
            next_line = list_file.readline()
            if not next_line:
                break
            img_list.append(next_line.rstrip())
        return img_list

    def __getitem__(self, index):
        data = dict((key, []) for key in self.input_modalities)
        img_name = os.path.splitext(os.path.basename(self.data_filenames[self.input_modalities[0]][index]))[0]
        for key in data:
            if key == 'normal' or key == 'rgb' or key == 'depth':
                data[key] = cv2.imread(self.data_filenames[key][index])
                data[key] = cv2.cvtColor(data[key], cv2.COLOR_BGR2RGB)
            else:
                data[key] = cv2.imread(self.data_filenames[key][index])
                data[key] = cv2.cvtColor(data[key], cv2.COLOR_BGR2GRAY)
                # append the channel dimension
                if key != 'labels':
                    data[key] = np.expand_dims(data[key], axis=2)
        data['labels'] = preprocess_mask(data['labels'])
        if self.transforms is not None:
            input_data = {k: data[k] for k in set(list(data.keys())) - set(['labels'])}
            input_data_keys = input_data.keys()
            # concatenate all the input data
            input_data = np.concatenate([input_data[k] for k in input_data_keys], axis=2)
            transformed = self.transforms(image = input_data, mask = data['labels'])
            input_data = transformed['image']
            mask = transformed['mask']
            current_index = 0
            for i, key in enumerate(input_data_keys):
                if key != 'curvature':
                    data[key] = input_data[current_index:current_index+3,:,:]
                    current_index = current_index+3
                else:
                    data[key] = input_data[current_index,:, :]
            data['labels'] = mask
            input_data = {k: data[k] for k in data.keys() - ['labels']}
        return img_name, input_data, data['labels'].long()

import glob 

class CrackSegmentationDataset(Dataset):
    def __init__(self, root_dir, num_classes, crack_dataset, image_set, input_modalities, sample_idx, transforms=None):
        self.n_class = num_classes
        self.crack_dataset = crack_dataset
        self.image_set = image_set
        self.root_dir = root_dir
        self.transforms = transforms
        self.modalities = ['rgb', 'labels']
        self.data_filenames = dict((key, []) for key in self.modalities)
        img_list = []
        img_list = self.get_image_list(self.root_dir, self.image_set, self.crack_dataset)
        for img_name in img_list:
            for modality in self.modalities:
                if os.path.isfile(os.path.join(root_dir, modality + '/{:s}'.format(img_name))):
                    self.data_filenames[modality].append(os.path.join(root_dir, image_set, modality, '{:s}'.format(img_name)))
        if self.image_set == 'train' and sample_idx is not None:
            self.data_filenames['rgb'] = [self.data_filenames['rgb'][i] for i in sample_idx]
            self.data_filenames['labels'] = [self.data_filenames['labels'][i] for i in sample_idx]

    def __len__(self):
        return len(self.data_filenames['rgb'])
    
    def get_image_list(self, root_dir, image_set, crack_dataset):
        if image_set == 'train':
            image_list = glob.glob(os.path.join(root_dir, 'train', 'labels', crack_dataset + '_' + '*.jpg'))
            image_list = [os.path.basename(x) for x in image_list]
        elif image_set == 'test':
            image_list = glob.glob(os.path.join(root_dir, 'test', 'rgb', crack_dataset + '_' + '*.jpg'))
            image_list = [os.path.basename(x) for x in image_list]
        return image_list
    
    def __getitem__(self, index):
        data = dict((key, []) for key in self.modalities)
        img_name = os.path.splitext(os.path.basename(self.data_filenames[self.modalities[0]][index]))[0]
        for key in data:
            if key == 'rgb':
                data[key] = cv2.imread(self.data_filenames[key][index])
                data[key] = cv2.cvtColor(data[key], cv2.COLOR_BGR2RGB)
            else:
                data[key] = cv2.imread(self.data_filenames[key][index])
                # Extra processing needed for labels :(
                if np.unique(data[key]) != np.array([0, 1]):
                    data[key][data[key] > 0] = 1
                assert np.unique(data[key])[1] == 1 and len(np.unique(data[key])) == 2, 'labels should be binary. Current label values: {}'.format(np.unique(data[key]))
                data[key] = cv2.cvtColor(data[key], cv2.COLOR_BGR2GRAY)
                # append the channel dimension
                if key != 'labels':
                    data[key] = np.expand_dims(data[key], axis=2)
        data['labels'] = preprocess_mask(data['labels'])
        if self.transforms is not None:
            input_data = {k: data[k] for k in set(list(data.keys())) - set(['labels'])}
            input_data_keys = input_data.keys()
            # concatenate all the input data
            input_data = np.concatenate([input_data[k] for k in input_data_keys], axis=2)
            transformed = self.transforms(image = input_data, mask = data['labels'])
            input_data = transformed['image']
            mask = transformed['mask']
            current_index = 0
            for i, key in enumerate(input_data_keys):
                if key != 'curvature':
                    data[key] = input_data[current_index:current_index+3,:,:]
                    current_index = current_index+3
                else:
                    data[key] = input_data[current_index,:, :]
            data['labels'] = mask
            input_data = {k: data[k] for k in data.keys() - ['labels']}
        # return input_data['rgb'], data['labels'].long() # This is for laplace redux package
        return img_name, input_data, data['labels'].long()

import collections
class camvidLoader():
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=None,
        transforms=None,
        img_norm=True,
        test_mode=False,
        sample_idx=None
    ):
        self.root = root
        self.split = split
        self.img_size = [224, 224]
        self.is_transform = is_transform
        self.transforms = transforms
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 12
        self.files = collections.defaultdict(list)

        if not self.test_mode:
            for split in ["train", "test", "val"]:
                file_list = os.listdir(root + "/" + split)
                self.files[split] = file_list
        if self.split == 'train' and sample_idx is not None:
            self.files['train'] = [self.files['train'][i] for i in sample_idx]

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + self.split + "/" + img_name
        lbl_path = self.root + "/" + self.split + "annot/" + img_name

        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = cv2.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY)

        lbl = preprocess_mask(lbl)
        if self.transforms is not None:
            transformed = self.transforms(image = img, mask = lbl)
            img = transformed['image']
            mask = transformed['mask']
        return img, mask.long()


if __name__ == '__main__':
    # Dataset parameters
    dataset = 'CrackForest'
    crop_size = 224
    root = '/~/HDD1/rrathnak/CAAP_Stereo/Datasets/CamVid'
    num_classes = 2

    data_transforms = A.Compose(
    [
        A.PadIfNeeded(min_height=crop_size, min_width=crop_size),
        A.RandomCrop(crop_size, crop_size),
        A.ShiftScaleRotate(shift_limit=0.055, scale_limit=0.05, rotate_limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )
    train_dataset = camvidLoader(root = root, is_transform=True, transforms=data_transforms, split = 'train', sample_idx = None)
    num_training = train_dataset.__len__()//2
    if num_training is not None:
        sample_idx = np.random.choice(train_dataset.__len__(), num_training, replace=False)
    else:
        sample_idx = None

    image_datasets = {x: camvidLoader(root = root, is_transform=True, transforms=data_transforms, split = x, sample_idx = sample_idx
    ) for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'val', 'test']}
    
    # get a batch of training data
    inputs, labels = next(iter(dataloaders['train']))