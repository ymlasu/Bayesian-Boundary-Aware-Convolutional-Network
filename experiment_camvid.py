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


from generate_class_weights import generate_class_weights, median_frequency_balancing
from torchmetrics import F1Score, Accuracy, PrecisionRecallCurve, CalibrationError

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

import torch
import torch.nn as nn
import time 

import numpy as np

def stochastic_log_softmax(mean, var, device):
	'''
	Returns Lx in the paper
	'''
	# Sample 't' times from a normal distribution to get x_hat
	x_hats = list()
	for t in range(10):
		eps = torch.randn(var.size()).to(device)
		x_hat = mean + torch.mul(var, eps)
		x_hats.append(x_hat)
	x_hats = torch.stack(x_hats)
	return torch.mean(torch.log_softmax(x_hats , dim = 2), dim = 0)	

def train(net, train_loader, optimizer, scheduler,
    num_classes, epoch, num_epochs, total_step,
    summary_logger, detailed_logger, device, sup_loss, 
    F1_score, F1_score_classwise, accuracy, accuracy_classwise,
    writer = None):
    batchwise_F1 = []
    batchwise_F1_classwise = []
    batchwise_accuracy = []
    batchwise_accuracy_classwise = []
    running_acc = 0
    running_loss = 0
    for iter, (input_data, target) in enumerate(train_loader):
        input_data = input_data.to(device)
        target = target.to(device) # target is a LongTensor
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out = net(input_data)
        y_predict = out[:, :num_classes, :, :]
        var = out[:, num_classes:, :, :]
        Lx = stochastic_log_softmax(y_predict, var, device)
        loss = sup_loss(Lx, target)
        loss.backward()
        optimizer.step()
        # compute the accuracy and f1 score for the current batch
        out_ = y_predict.detach().clone()
        acc = accuracy(out_.detach().cpu(), target.cpu())
        acc_classwise = accuracy_classwise(out_.detach().cpu(), target.cpu())
        f1_score = F1_score(out_.detach().cpu(), target.cpu()).numpy()
        f1_score_classwise = F1_score_classwise(out_.detach().cpu(), target.cpu()).numpy()
        batchwise_F1.append(f1_score)
        batchwise_F1_classwise.append(f1_score_classwise)
        batchwise_accuracy.append(acc)
        batchwise_accuracy_classwise.append(acc_classwise)
        running_acc += acc
        running_loss += loss.item()
        # log the loss and accuracy for the current batch
        detailed_logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, F1 Score: {:.4f}, Classwise F1: {}'
        .format(epoch+1, num_epochs, iter+1, total_step, loss.item(), acc, f1_score, f1_score_classwise))
    if optimizer == 'SGD':
        scheduler.step()
    # print the loss and accuracy for the current epoch
    epoch_acc = running_acc/(iter+1)
    epoch_loss = running_loss/(iter + 1)
    epoch_F1 = np.mean(batchwise_F1)
    epoch_classwise_F1 = np.mean(batchwise_F1_classwise, axis = 0)
    if (iter + 1) % 10 == 0:
        summary_logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, F1 Score: {:.4f}'.format(epoch + 1, num_epochs, iter + 1, total_step, loss.item(), epoch_acc, epoch_F1))
        # Log classwise F1 scores
        # summary_logger.info('Classwise F1 Scores: {}'.format(epoch_classwise_F1))
    if writer is not None:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Train/F1 Score', epoch_F1, epoch)
        writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    return epoch_loss, epoch_F1, epoch_classwise_F1

def validate(net, val_loader, num_classes,
    epoch, num_epochs, total_step,
    summary_logger, detailed_logger, device, sup_loss, 
    F1_score, F1_score_classwise, accuracy, accuracy_classwise, 
    writer = None):
    batch_F1 = []
    batch_F1_classwise = []
    batch_accuracy = []
    batch_accuracy_classwise = []
    running_acc = 0
    running_loss = 0
    with torch.no_grad():
        # Iterate over the validation data and generate predictions
        for iter, (input_data, target) in enumerate(val_loader):
            input_data = input_data.to(device)
            target = target.to(device)
            out = net(input_data)
            y_predict = out[:, :num_classes, :, :]
            var = out[:, num_classes:, :, :]
            Lx = stochastic_log_softmax(y_predict, var, device)
            loss = sup_loss(Lx, target)
            # compute the accuracy and f1 score for the current batch
            out_ = y_predict.detach().clone()
            acc = accuracy(out_.detach().cpu(), target.cpu())
            acc_classwise = accuracy_classwise(out_.detach().cpu(), target.cpu())
            f1_score = F1_score(out_.detach().cpu(), target.cpu()).numpy()
            f1_score_classwise = F1_score_classwise(out_.detach().cpu(), target.cpu()).numpy()
            batch_F1.append(f1_score)
            batch_F1_classwise.append(f1_score_classwise)
            batch_accuracy.append(acc)
            batch_accuracy_classwise.append(acc_classwise)
            running_acc += acc
            running_loss += loss.item()
            # log the loss and accuracy for the current batch
            detailed_logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, F1 Score: {:.4f}, Classwise F1: {}'
            .format(epoch+1, num_epochs, iter+1, total_step, loss.item(), acc, f1_score, f1_score_classwise))
    epoch_acc = running_acc/(iter+1)
    epoch_loss = running_loss/(iter + 1)
    epoch_F1 = np.mean(batch_F1)
    epoch_classwise_F1 = np.mean(batch_F1_classwise, axis = 0)
    if (iter + 1) % 10 == 0:
        summary_logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, F1 Score: {:.4f}'.format(epoch + 1, num_epochs, iter + 1, total_step, loss.item(), epoch_acc, epoch_F1))
        # Log classwise F1 scores
        # summary_logger.info('Classwise F1 Scores: {}'.format(epoch_classwise_F1))
    if writer is not None:
        writer.add_scalar('Validation/Loss', epoch_loss, epoch)
        writer.add_scalar('Validation/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Validation/F1 Score', epoch_F1, epoch)
    return epoch_loss, epoch_acc, epoch_F1, epoch_classwise_F1

def test(net, test_loader, num_classes, mc_samples, train_len,
    summary_logger, device, 
    F1_score, F1_score_classwise, accuracy, accuracy_classwise):
    # TODO: Modify varnames to batchwise, reorganize inner loop according to 
    # the new varnames, and according to analyze_model.ipynb
    batch_F1 = []
    batch_F1_classwise = []
    batch_accuracy = []
    batch_accuracy_classwise = []
    batch_epistemic = []
    batch_epistemic_classwise = []
    batch_aleatoric = []
    batch_aleatoric_classwise = []
    softmax = nn.Softmax(dim = 1)
    with torch.no_grad():
        for iter, (input_data, target) in enumerate(test_loader):
            sampled_outs = []
            means = []
            means_noSoftmax = []
            sample_F1 = []
            sample_F1_classwise = []
            sample_accuracy = []
            sample_accuracy_classwise = []
            input_data = input_data.to(device)
            target = target.to(device)
            # Predicted aleatoric variance from a single pass
            aleatoric_uncertainty = net(input_data)[:, num_classes:, :, :]
            assert aleatoric_uncertainty.shape[1] == num_classes, "Aleatoric uncertainty shape error."
            aleatoric_uncertainty = np.exp(aleatoric_uncertainty.detach().clone().cpu().numpy())
            for i in range(mc_samples):
                sampled_outs.append(net(input_data))
            for out in sampled_outs:
                N, C, H, W = out.shape
                mean = out[:, :num_classes, :, :]
                means_noSoftmax.append(mean.detach().clone().cpu().numpy())
                mean = softmax(mean)
                assert torch.allclose(mean.sum(dim = 1), torch.ones(N, H, W).to(device)), "AssertionError: Probabilities do not sum to 1"
                means.append(mean.detach().clone().cpu().numpy())
                f1_score = F1_score(mean.detach().cpu(), target.cpu()).numpy()
                f1_score_ = F1_score(out[:, :num_classes, :, :].detach().cpu(), target.cpu()).numpy()
                assert np.allclose(f1_score, f1_score_), "AssertionError: F1 score is not the same between softmax mean and out"
                f1_score_classwise = F1_score_classwise(mean.detach().cpu(), target.cpu()).numpy()
                acc = accuracy(mean.detach().cpu(), target.cpu()).numpy()
                acc_classwise = accuracy_classwise(mean.detach().cpu(), target.cpu()).numpy()
                sample_F1.append(f1_score)
                sample_F1_classwise.append(f1_score_classwise)
                sample_accuracy.append(acc)
                sample_accuracy_classwise.append(acc_classwise)
            # compute the accuracy and f1 score for the current batch
            batch_F1.append(np.mean(sample_F1, axis = 0))
            batch_F1_classwise.append(np.mean(sample_F1_classwise, axis = 0))
            batch_accuracy.append(np.mean(sample_accuracy))
            batch_accuracy_classwise.append(np.mean(sample_accuracy_classwise, axis = 0))
            batch_mean = np.mean(np.stack(means), axis = 0)
            N, C, H, W = batch_mean.shape
            pred = batch_mean.transpose(0, 2, 3, 1).reshape(-1, num_classes).argmax(axis = 1).reshape(N, H, W)
            epistemic_uncertainty = np.var(np.stack(means_noSoftmax), axis = 0)
            classwise_epistemic_uncertainty = np.mean(epistemic_uncertainty, axis = (0,2,3))
            classwise_aleatoric_uncertainty = np.mean(aleatoric_uncertainty, axis = (0,2,3))
            batch_epistemic.append(np.mean(epistemic_uncertainty))
            batch_epistemic_classwise.append(classwise_epistemic_uncertainty)
            batch_aleatoric.append(np.mean(aleatoric_uncertainty))
            batch_aleatoric_classwise.append(classwise_aleatoric_uncertainty)
    # compute the average accuracy and f1 score for the entire epoch
    epoch_acc = np.mean(batch_accuracy)
    epoch_F1 = np.mean(batch_F1)
    epoch_classwise_F1 = np.mean(batch_F1_classwise, axis = 0)
    epoch_classwise_accuracy = np.mean(batch_accuracy_classwise, axis = 0)
    epoch_epistemic = np.mean(batch_epistemic)
    epoch_epistemic_classwise = np.mean(batch_epistemic_classwise, axis = 0)
    epoch_aleatoric = np.mean(batch_aleatoric)
    epoch_aleatoric_classwise = np.mean(batch_aleatoric_classwise, axis = 0)
    if summary_logger is not None:
        summary_logger.info('Training Set Length {}, Accuracy: {:.4f}, F1 Score: {:.4f}'.format(train_len, epoch_acc, epoch_F1))
    return epoch_F1, epoch_classwise_F1, epoch_acc, epoch_classwise_accuracy, epoch_epistemic, epoch_epistemic_classwise, epoch_aleatoric, epoch_aleatoric_classwise                        



# hyperparameters for training
num_epochs = 300
batch_size = 48
optimizer = 'SGD'
learning_rate = 0.1
optim_weight_decay = 1e-5
momentum = 0.9
step_size = 50
gamma = 0.5
checkpoint_freq = 10
dataparallel = True
class_balancing_method = 'median' # 'precalculated' or 'balanced' or None

# Dataset parameters
dataset = 'CamVid'
crop_size = 224
root_dir = "/~/HDD1/rrathnak/CAAP_Stereo/Datasets/CamVid"
num_classes = 12
num_training = 275 # Set to None for all training data

# network parameters
backbone = 'resnet50'
pretrained = True
p = 0.5

# uncertainty methods for evaluation
uncertainty_methods = ['entropy', 'variance', 'bald', 'margin', 'random']

# MC Dropout parameters
mc_samples = 25

# set date to current date
import datetime
date = datetime.datetime.now().strftime("%Y-%m-%d")

# Directories
model_dir = '/~/HDD1/rrathnak/CAAP_Stereo/MCDROPOUT_EXPERIMENTS/models'
if num_training is None:
    training_samples = 'all'
else:
    training_samples = str(num_training)
directory = os.path.join(model_dir, date + '_' + backbone +'_' + ''.join(str(d) for d in dataset) + '_' + str(training_samples) + 'Train')
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
        A.Normalize(mean=(0.39068785, 0.40521392, 0.41434407), std=(0.29652068, 0.30514979, 0.30080369)),
        ToTensorV2()
    ]
)
from dataset import camvidLoader
train_dataset = camvidLoader(root = root_dir, is_transform=True, transforms=data_transforms, split = 'train', sample_idx = None)
if num_training is not None:
    sample_idx = np.random.choice(train_dataset.__len__(), num_training, replace=False)
else:
    sample_idx = None

image_datasets = {x: camvidLoader(root = root_dir, is_transform=True, transforms=data_transforms, split = x, sample_idx = sample_idx
) for x in ['train', 'val', 'test']}


dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create net
net = SegmentationNetwork(backbone = backbone, pretrained = pretrained, p = p, num_classes = num_classes)
if dataparallel:
    net = nn.DataParallel(net, device_ids = [0,1,2,3,4,5,6,7])
net.to(device)

# Loss and optimizer
# # Generate class weights for training set
if class_balancing_method == 'precalculated':
    weight = torch.tensor([0.5248, 10.5860]).type(torch.FloatTensor).to(device)
elif class_balancing_method == 'balanced':
    class_series = []
    for iter, (data, target) in enumerate(dataloader['train']):
        class_series.append(target.detach().numpy())
    class_series = np.vstack(class_series).reshape(-1)
    class_weights = generate_class_weights(class_series)
    weight = torch.tensor(list(class_weights.values())).type(torch.FloatTensor).to(device)
elif class_balancing_method == 'median':
    class_series = []
    for iter, (data, target) in enumerate(dataloader['train']):
        class_series.append(target.detach().numpy())
    class_series = np.vstack(class_series).reshape(-1)
    class_weights = median_frequency_balancing(class_series)
    weight = torch.tensor(class_weights).type(torch.FloatTensor).to(device)
else:
    weight = None
logging.info('CLASS WEIGHTS %s' % weight)
sup_loss = nn.NLLLoss(weight = weight).to(device)
if optimizer == 'SGD':
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=optim_weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
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
            _, _, _ = train(
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
                device = device,
                sup_loss = sup_loss,
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
                val_loader = dataloader['val'],
                num_classes = num_classes,
                epoch = epoch,
                num_epochs = num_epochs,
                total_step = total_step,
                summary_logger = logger,
                detailed_logger = detailed_logger,
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
# net.load_state_dict(best_net_wts)
# net.eval()
# if dataparallel == True:
#     net.module.dropout.train() # Turn on MC Dropout
# else:
#     net.dropout.train()
# epoch_F1, epoch_classwise_F1, epoch_acc, epoch_classwise_accuracy, epoch_epistemic, epoch_epistemic_classwise, epoch_aleatoric, epoch_aleatoric_classwise = test(
#     net = net,
#     test_loader = dataloader['test'],
#     num_classes = num_classes,
#     mc_samples = mc_samples,
#     train_len = image_datasets['train'].__len__(),
#     summary_logger = logger,
#     device = device,
#     F1_score = F1_score,
#     F1_score_classwise = F1_score_classwise,
#     accuracy = accuracy,
#     accuracy_classwise = accuracy_classwise)
# row = [
#     len(dataloader['train']),
#     epoch_F1, epoch_classwise_F1[0], epoch_classwise_F1[1],
#     epoch_epistemic, epoch_epistemic_classwise[0], epoch_epistemic_classwise[1],
#     epoch_aleatoric, epoch_aleatoric_classwise[0], epoch_aleatoric_classwise[1]
# ]
# csvwriter.writerow(row)
# csvfile.close()