'''
    Try to use the idea of image classification for face recognition.
    Just for learning image classification, it is not good and useful for face recognition.
'''
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from backbone.resnet import resnet34
from dataset.dataset import Dataset

cudnn.benchmark = True

parse = argparse.ArgumentParser(description='face classification')
parse.add_argument('--train_root', type=str, default='../dataset/lfw-align-112x112')
# parse.add_argument('--train_root', type=str, default='E:\server\competition\multi_race_face_recognition\public_dataset/lfw-align-112x112')
parse.add_argument('--train_list', type=str, default='../dataset/dataset_list/lfw_train_list.txt')
parse.add_argument('--pretrained_model_path', type=str, default='../pretrained_model/resnet34/resnet34.pth')
parse.add_argument('--img_mode', type=str, default='RGB')
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--k', type=int, default=10, help='the time of fc layer lr')
parse.add_argument('--weight_decay', type=float, default=5e-4)
parse.add_argument('--num_classes', type=int, default=5749)
parse.add_argument('--batch_size', type=int, default=64)
parse.add_argument('--max_epoch', type=int, default=40)
parse.add_argument('--optimizer', type=str, default='sgd')
parse.add_argument('--criterion', type=str, default='CrossEntropyLoss')
parse.add_argument('--scheduler', type=str, default='MultiStep')
parse.add_argument('--milestones', type=list, default=[7,14,21,28])
parse.add_argument('--gamma', type=float, default=0.5)
parse.add_argument('--use_gpu', type=bool, default=True)
parse.add_argument('--pin_memory', type=bool, default=True)
parse.add_argument('--parallel', type=bool, default=None)
args = parse.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

def get_model():
    model = resnet34(pretrained=False, progress=False)
    model.load_state_dict(torch.load(args.pretrained_model_path))
    model.fc = nn.Linear(512, args.num_classes)
    return model

def get_criterion():
    if args.criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    return criterion

def get_optimizer(model):
    conv_layer = []
    fc_layer = []
    for key, parameter in model.named_parameters():
        if 'fc' not in key:
            conv_layer += [parameter]
        else:
            fc_layer += [parameter]
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params':conv_layer, 'lr':args.lr},
                                 {'params':fc_layer, 'lr':args.k*args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def get_scheduler(optimizer):
    if args.scheduler == 'MultiStep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    return scheduler

def save_checkpoint(model, epoch, loss, acc):
    checkpoint_dir = '../pretrained_model/resnet34/checkpoint/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = checkpoint_dir+'%d_%.4f_%.4f' % (epoch,loss,acc)
    if args.parallel == True:
        torch.save(model.module.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)

def train():
    # dataset and dataloader
    dataset = Dataset(root = args.train_root,
                      dataset_list = args.train_list,
                      phase='train',
                      input_shape=(112, 112),
                      img_mode = 'RGB')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=args.pin_memory, shuffle=True)
    # model
    model = get_model()
    model.to(device)
    # parallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        args.parallel = True
    # criterion
    criterion = get_criterion()
    # optimizer
    optimizer = get_optimizer(model)
    # scheduler
    scheduler = get_scheduler(optimizer)

    # training
    for epoch in range(args.max_epoch):
        model.train()
        tbar = tqdm(dataloader)
        loss_avg = 0.0
        acc_avg = 0.0
        for idx, (data_input, label) in enumerate(tbar):
            data_input = data_input.to(device)
            label = label.to(device).long()
            # predict
            output = model(data_input)
            # compute loss
            loss = criterion(output, label)
            # clean the optimizer gradient buffer cache
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update weight parameters
            optimizer.step()
            # count the training result accuracy
            output = output.data.cpu().numpy()
            pre_label = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean(pre_label==label)
            tbar.set_description('epoch:%d, loss:%.4f, acc:%.4f' %(epoch, loss.item(), acc))
            # record the loss and acc
            loss_avg += loss.item()
            acc_avg += acc
        loss_avg /= len(dataloader)
        acc_avg /= len(dataloader)
        # save model
        if (epoch+1) % 2 == 0:
            save_checkpoint(model, epoch, loss_avg, acc_avg)
        scheduler.step()

if __name__ == '__main__':
    train()