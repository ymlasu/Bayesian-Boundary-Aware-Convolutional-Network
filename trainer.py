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
    summary_logger, detailed_logger, input_modalities, device, sup_loss, 
    F1_score, F1_score_classwise, accuracy, accuracy_classwise,
    writer = None):
    batchwise_F1 = []
    batchwise_F1_classwise = []
    batchwise_accuracy = []
    batchwise_accuracy_classwise = []
    running_acc = 0
    running_loss = 0
    for iter, (img_name, input_data, target) in enumerate(train_loader):
        for modality in input_modalities:
            input_data[modality] = input_data[modality].to(device)
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
        summary_logger.info('Classwise F1 Scores: {}'.format(epoch_classwise_F1))
    if writer is not None:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Train/F1 Score', epoch_F1, epoch)
        writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    return epoch_loss, epoch_F1, epoch_classwise_F1


def train_abl(net, train_loader, optimizer, scheduler,
    num_classes, epoch, num_epochs, total_step,
    summary_logger, detailed_logger, input_modalities, device, sup_loss, abl_loss, iou_loss,
    F1_score, F1_score_classwise, accuracy, accuracy_classwise,
    writer = None):
    batchwise_F1 = []
    batchwise_F1_classwise = []
    batchwise_accuracy = []
    batchwise_accuracy_classwise = []
    running_acc = 0
    running_loss = 0
    for iter, (img_name, input_data, target) in enumerate(train_loader):
        for modality in input_modalities:
            input_data[modality] = input_data[modality].to(device)
        target = target.to(device) # target is a LongTensor
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out = net(input_data)
        y_predict = out[:, :num_classes, :, :]
        var = out[:, num_classes:, :, :]
        Lx = stochastic_log_softmax(y_predict, var, device)
        loss_1 = sup_loss(Lx, target)
        loss_2 = abl_loss(Lx, target)
        loss_3, _ = iou_loss(Lx, target)
        loss = loss_1 + loss_2 + loss_3
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
        summary_logger.info('Classwise F1 Scores: {}'.format(epoch_classwise_F1))
    if writer is not None:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Train/F1 Score', epoch_F1, epoch)
        writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    return epoch_loss, epoch_F1, epoch_classwise_F1


def train_contrast_abl(net, train_loader, optimizer, scheduler,
    num_classes, epoch, num_epochs, total_step,
    summary_logger, detailed_logger, input_modalities, device, sup_loss, abl_loss, iou_loss, contrast_loss,
    F1_score, F1_score_classwise, accuracy, accuracy_classwise,
    writer = None):
    batchwise_F1 = []
    batchwise_F1_classwise = []
    batchwise_accuracy = []
    batchwise_accuracy_classwise = []
    running_acc = 0
    running_loss = 0
    for iter, (img_name, input_data, target) in enumerate(train_loader):
        for modality in input_modalities:
            input_data[modality] = input_data[modality].to(device)
        target = target.to(device) # target is a LongTensor
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out = net(input_data)
        y_predict = out[:, :num_classes, :, :]
        var = out[:, num_classes:, :, :]
        Lx = stochastic_log_softmax(y_predict, var, device)
        loss_1 = sup_loss(Lx, target)
        loss_2 = abl_loss(Lx, target)
        loss_3, _ = iou_loss(Lx, target)
        loss_4 = contrast_loss(Lx, target)
        loss = loss_1 + loss_2 + loss_3
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
        summary_logger.info('Classwise F1 Scores: {}'.format(epoch_classwise_F1))
    if writer is not None:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Train/F1 Score', epoch_F1, epoch)
        writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    return epoch_loss, epoch_F1, epoch_classwise_F1


def train_swag(net, swa_net, train_loader, optimizer, scheduler, swa_scheduler, swa_start,
    num_classes, epoch, num_epochs, total_step,
    summary_logger, detailed_logger, input_modalities, device, sup_loss, abl_loss, iou_loss,
    F1_score, F1_score_classwise, accuracy, accuracy_classwise,
    writer = None
):
    batchwise_F1 = []
    batchwise_F1_classwise = []
    batchwise_accuracy = []
    batchwise_accuracy_classwise = []
    running_acc = 0
    running_loss = 0
    for iter, (img_name, input_data, target) in enumerate(train_loader):
        for modality in input_modalities:
            input_data[modality] = input_data[modality].to(device)
        target = target.to(device) # target is a LongTensor
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out = net(input_data)
        y_predict = out[:, :num_classes, :, :]
        var = out[:, num_classes:, :, :]
        Lx = stochastic_log_softmax(y_predict, var, device)
        loss_1 = sup_loss(Lx, target)
        loss_2 = abl_loss(Lx, target)
        loss_3, _ = iou_loss(Lx, target)
        loss = loss_1 + loss_2 + loss_3
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
    if epoch > swa_start:
        swa_net.update_parameters(net)
        swa_scheduler.step()
    else:
        scheduler.step()
    # print the loss and accuracy for the current epoch
    epoch_acc = running_acc/(iter+1)
    epoch_loss = running_loss/(iter + 1)
    epoch_F1 = np.mean(batchwise_F1)
    epoch_classwise_F1 = np.mean(batchwise_F1_classwise, axis = 0)
    if (iter + 1) % 10 == 0:
        summary_logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, F1 Score: {:.4f}'.format(epoch + 1, num_epochs, iter + 1, total_step, loss.item(), epoch_acc, epoch_F1))
        # Log classwise F1 scores
        summary_logger.info('Classwise F1 Scores: {}'.format(epoch_classwise_F1))
    if writer is not None:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Train/F1 Score', epoch_F1, epoch)
        writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    return epoch_loss, epoch_F1, epoch_classwise_F1

def validate(net, val_loader, num_classes,
    epoch, num_epochs, total_step,
    summary_logger, detailed_logger, input_modalities, device, sup_loss, 
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
        for iter, (img_name, input_data, target) in enumerate(val_loader):
            for modality in input_modalities:
                input_data[modality] = input_data[modality].to(device)
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
        summary_logger.info('Classwise F1 Scores: {}'.format(epoch_classwise_F1))
    if writer is not None:
        writer.add_scalar('Validation/Loss', epoch_loss, epoch)
        writer.add_scalar('Validation/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Validation/F1 Score', epoch_F1, epoch)
    return epoch_loss, epoch_acc, epoch_F1, epoch_classwise_F1

def test(net, test_loader, num_classes, mc_samples, train_len,
    summary_logger, input_modalities, device, 
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
        for iter, (img_name, input_data, target) in enumerate(test_loader):
            sampled_outs = []
            means = []
            means_noSoftmax = []
            sample_F1 = []
            sample_F1_classwise = []
            sample_accuracy = []
            sample_accuracy_classwise = []
            for modality in input_modalities:
                input_data[modality] = input_data[modality].to(device)
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

