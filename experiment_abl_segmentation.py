# Segmentation network for a small dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision import datasets
from torchvision.utils import save_image
from torchvision import models

import numpy as np
import matplotlib.pyplot as plt

import os
import logging
import argparse
import copy 
import csv

from network import *
from dataset import *
from trainer import *
from generate_class_weights import generate_class_weights
from torchmetrics import F1Score, Accuracy, PrecisionRecallCurve, CalibrationError

from activeBoundaryLoss import ABL
from lovasz_softmax import LovaszSoftmaxV1
# Tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


seed = 1
# set random seed for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s")

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

import argparse 

parser = argparse.ArgumentParser(description='PyTorch MCDropout Training')
parser.add_argument('--num_training', type=int, default=None, help='Number of training samples')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
args = parser.parse_args()


# hyperparameters for training
num_epochs = 500
batch_size = args.batch_size
optimizer = 'SGD'
learning_rate = 1e-2
optim_weight_decay = 1e-5
momentum = 0.9
step_size = 100
gamma = 0.5
checkpoint_freq = 10
dataparallel = False
class_weights = 'precalculated' # 'precalculated' or 'balanced' or None
gpu_id = 'cuda:' + str(args.gpu_id)
# Dataset parameters
dataset = 'DeepCrack'
crop_size = 224
root_dir = '/~/HDD1/rrathnak/CAAP_Stereo/Datasets/crack_segmentation_dataset'
num_classes = 2
input_modalities = ['rgb']
num_training = args.num_training # Set to None for all training data

# network parameters
backbone = 'resnet50'
pretrained = True
p = args.dropout_ratio

# uncertainty methods for evaluation
uncertainty_methods = ['entropy', 'variance', 'bald', 'margin', 'random']

# MC Dropout parameters
mc_samples = 25

# Early stopping parameters
patience = 10
wait = 0
early_stop = True


# set date to current date
import datetime
date = datetime.datetime.now().strftime("%Y-%m-%d")

# Directories
model_dir = '/~/HDD1/rrathnak/CAAP_Stereo/MCDROPOUT_EXPERIMENTS/models'
if num_training is None:
    training_samples = 'all'
else:
    training_samples = str(num_training)
directory = os.path.join(model_dir, date + '_' + backbone +'_' + ''.join(str(d) for d in dataset) + '_' + str(training_samples) + 'Train' + '_' + str(p) + 'Dropout' + '_ABLIOUSegLoss')
if not os.path.exists(directory):
    os.makedirs(directory)
log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
tensorboard_dir = os.path.join(directory, 'tensorboard')
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
checkpoint_dir = os.path.join(directory, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
# initialize logger
logger = setup_logger('summary_logger', log_dir + '/summaryLog.log')
detailed_logger = setup_logger('detailed_logger', log_dir + '/detailed.log')
# Import dataset and create dataloader
import albumentations as A
from albumentations.pytorch import ToTensorV2

data_transforms = A.Compose(
    [
        A.PadIfNeeded(min_height=crop_size, min_width=crop_size),
        A.RandomCrop(crop_size, crop_size),
        A.ShiftScaleRotate(shift_limit=0.055, scale_limit=0.05, rotate_limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(crop_size, crop_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)
# Import dataset, select training samples according to num_training
from dataset import CrackSegmentationDataset
train_dataset = CrackSegmentationDataset(root_dir=root_dir, num_classes = num_classes, crack_dataset = dataset, image_set='train', transforms=data_transforms, sample_idx = None, input_modalities=input_modalities)
np.random.seed(seed)
if num_training is not None:
    sample_idx = np.random.choice(train_dataset.__len__(), num_training, replace=False)
else:
    sample_idx = None
image_datasets = {x: CrackSegmentationDataset(root_dir = root_dir, num_classes = num_classes, crack_dataset = dataset, 
transforms = data_transforms, image_set = x, input_modalities = input_modalities, sample_idx = sample_idx
) for x in ['train', 'test']}

# Create dataloaders
dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

# Device configuration
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')

# Create net
net = SegmentationNetwork(backbone = backbone, pretrained = pretrained, p = p, num_classes = num_classes)
if dataparallel:
    net = nn.DataParallel(net, device_ids = [1,2,3,4,5,6])
net.to(device)

# Loss and optimizer
# # Generate class weights for training set
if class_weights == 'precalculated':
    weight = torch.tensor([0.5248, 10.5860]).type(torch.FloatTensor).to(device)
elif class_weights == 'balanced':
    class_series = []
    for iter, (img_name, data, target) in enumerate(dataloader['train']):
        class_series.append(target.detach().numpy())
    class_series = np.vstack(class_series).reshape(-1)
    class_weights = generate_class_weights(class_series)
    weight = torch.tensor(list(class_weights.values())).type(torch.FloatTensor).to(device)
else:
    weight = None
logging.info('CLASS WEIGHTS %s' % weight)
sup_loss = nn.NLLLoss(weight = weight).to(device)
abl_loss = ABL(device = device)
iou_loss = LovaszSoftmaxV1(reduction='mean', ignore_index=255).to(device)
if optimizer == 'SGD':
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=optim_weight_decay)
    # Poly learning rate scheduler - (1- iter/max_iter)^power
    power = 0.9
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (1 - x / (len(dataloader['train']) * num_epochs)) ** power)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
elif optimizer == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=optim_weight_decay)

# Metrics for evaluation
F1_score = F1Score(mdmc_average='samplewise', average = 'macro',num_classes=num_classes)
F1_score_classwise = F1Score(average = 'none', num_classes = num_classes, mdmc_average='samplewise')
accuracy = Accuracy(mdmc_average='samplewise', num_classes = num_classes, average = 'macro')
accuracy_classwise = Accuracy(average = 'none', num_classes = num_classes, mdmc_average='samplewise')

best_net_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0
best_epoch = 0
best_loss = 1000
best_f1 = 0.0

csvfile = open(directory + '/' + 'results.csv', 'w')
fields = ['TrainSamples',
'MeanF1', 'Class0_F1', 'Class1_F1', 
'MeanEpi', 'Class0_Epi', 'Class1_Epi', 
'MeanAle', 'Class0_Ale', 'Class1_Ale']
csvwriter = csv.writer(csvfile)
csvwriter.writerow(fields)
exclude_to_net_key = 'labels'

total_step = len(dataloader['train'])

# log all hyperparameters
logging.info('Hyperparameters for training')
logging.info('Batch size: %s' % batch_size)
logging.info('Number of epochs: %s' % num_epochs)
logging.info('Learning rate: %s' % learning_rate)
logging.info('Momentum: %s' % momentum)
logging.info('Step size: %s' % step_size)
logging.info('Gamma: %s' % gamma)
logging.info('Optimizer: %s' % optimizer)
logging.info('Loss: %s' % sup_loss)
logging.info('Class weights: %s' % weight)

# log all dataset parameters
logging.info('Dataset parameters')
logging.info('Dataset: %s' % dataset)
logging.info('Number of training samples: %s' % dataset_sizes['train'])
logging.info('Number of test samples: %s' % dataset_sizes['test'])
logging.info('Number of classes: %s' % num_classes)
logging.info('Input modalities: %s' % input_modalities)
logging.info('Crop size: %s' % crop_size)
logging.info('Data augmentation: %s' % data_transforms)

# network parameters
logging.info('Network parameters')
logging.info('Backbone: %s' % backbone)
logging.info('Pretrained: %s' % pretrained)
logging.info('Dropout: %s' % p)
logging.info('MC Samples: %s' % mc_samples)

# Train-Val loop
for epoch in range(num_epochs):
    detailed_logger.info('Epoch {}/{}'.format(epoch, num_epochs))
    for phase in ['train', 'test']:
        running_acc = 0.0
        running_loss = 0.0
        detailed_logger.info("Current phase: %s" % phase)
        if phase == 'train':
            net.train() # Fine-tune the classifier
            _, _, _ = train_abl(
                net = net,
                writer = writer,
                train_loader = dataloader['train'],
                optimizer = optimizer,
                scheduler = scheduler,
                num_classes = num_classes,
                epoch = epoch,
                num_epochs = num_epochs,
                total_step = total_step,
                summary_logger = logger,
                detailed_logger = detailed_logger,
                input_modalities = input_modalities,
                device = device,
                sup_loss = sup_loss,
                abl_loss = abl_loss,
                iou_loss = iou_loss,
                F1_score = F1_score,
                F1_score_classwise = F1_score_classwise,
                accuracy = accuracy,
                accuracy_classwise = accuracy_classwise
            )
        else:
            net.eval()
            if dataparallel == True:
                net.module.dropout.train() # Turn on MC Dropout
            else:
                net.dropout.train()
            epoch_loss, epoch_acc, epoch_F1, epoch_classwise_F1 = validate(
                net = net,
                writer = writer,
                val_loader = dataloader['test'],
                num_classes = num_classes,
                epoch = epoch,
                num_epochs = num_epochs,
                total_step = total_step,
                summary_logger = logger,
                detailed_logger = detailed_logger,
                input_modalities = input_modalities,
                device = device,
                sup_loss = sup_loss,
                F1_score = F1_score,
                F1_score_classwise = F1_score_classwise,
                accuracy = accuracy,
                accuracy_classwise = accuracy_classwise
            )
            # Save best model
            if epoch_F1 > best_f1:
                best_f1 = epoch_F1
                best_epoch = epoch + 1
                best_net_wts = copy.deepcopy(net.state_dict())
                best_loss = epoch_loss
                best_acc_classwise = accuracy_classwise
                best_f1_classwise = F1_score_classwise
                torch.save(best_net_wts, os.path.join(best_dir, 'best_model.pth'))
                logger.info("Best model saved!")
                logger.info("Best F1: %f" % best_f1)
                logger.info("Best loss: %f" % best_loss)
                logger.info("Best epoch: %d" % best_epoch)
                logger.info("Best accuracy classwise: %s" % str(best_acc_classwise))
                logger.info("Best F1 classwise: %s" % str(epoch_classwise_F1))
            # Save checkpoint model
            if epoch % checkpoint_freq == 0:
                torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'checkpoint.pth'))
                logger.info("Checkpoint model saved at epoch %d" % epoch)

# Save the final model
torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'final_model.pth'))
logger.info("Final model saved at epoch %d" % epoch)

# Evaluate the best model in the test set
net.load_state_dict(best_net_wts)
net.eval()
if dataparallel == True:
    net.module.dropout.train() # Turn on MC Dropout
else:
    net.dropout.train()
epoch_F1, epoch_classwise_F1, epoch_acc, epoch_classwise_accuracy, epoch_epistemic, epoch_epistemic_classwise, epoch_aleatoric, epoch_aleatoric_classwise = test(
    net = net,
    test_loader = dataloader['test'],
    num_classes = num_classes,
    mc_samples = mc_samples,
    train_len = image_datasets['train'].__len__(),
    summary_logger = logger,
    input_modalities = input_modalities,
    device = device,
    F1_score = F1_score,
    F1_score_classwise = F1_score_classwise,
    accuracy = accuracy,
    accuracy_classwise = accuracy_classwise)
row = [
    len(dataloader['train']),
    epoch_F1, epoch_classwise_F1[0], epoch_classwise_F1[1],
    epoch_epistemic, epoch_epistemic_classwise[0], epoch_epistemic_classwise[1],
    epoch_aleatoric, epoch_aleatoric_classwise[0], epoch_aleatoric_classwise[1]
]
csvwriter.writerow(row)
csvfile.close()
