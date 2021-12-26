'''
    The lfw dataset is usually used as a test dataset. But our experiment doesn't has long time and better hardware,
    so we use this dataset to learn training a face recognition neural network as a teaching case.
'''
import os
import time
import json
import numpy as np
from tqdm import tqdm
from config.config import args
from lfw_test import lfw_test
from dataset.dataset import Dataset
from backbone.mobilefacenet import MobileFacenet
from backbone.utils import get_layers, prepare_model, subnet_to_dense
from head.metrics import ArcMarginProduct, AddMarginProduct, SphereProduct
from lossfunction.focal_loss import FocalLoss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def save_model(model, metric_fc, iterations, val_acc):
    # save config parameter
    config_path = os.path.join(args.save_path, 'config', 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    # save model.state_dict()
    model_save_path = os.path.join(args.save_path, 'model', '%d_%.4f.pth' % (iterations, val_acc))
    metric_save_path = os.path.join(args.save_path, 'metric', '%d_%.4f.pth' % (iterations, val_acc))

    dense_dict = subnet_to_dense(model.state_dict(), args.k)

    if args.parallel == True:
        torch.save(model.module.state_dict(), model_save_path)
        torch.save(metric_fc.module.state_dict(), metric_save_path)
    else:
        torch.save(dense_dict, model_save_path)
        torch.save(metric_fc.state_dict(), metric_save_path)

    # save checkpoint
    checkpoint = os.path.join(args.save_path, 'checkpoint.tar')
    state_dict= {
        'model': model.module.state_dict() if args.parallel else model.state_dict(),
        'metric': metric_fc.module.state_dict() if args.parallel else metric_fc.state_dict(),
        'optimizer': optimizer.state_dict(),
        'schduler': scheduler.state_dict(),
        'criterion': criterion.state_dict(),
        'parameter': args.__dict__,
        'iterations': iterations,
    }
    torch.save(state_dict, checkpoint)
def get_model(args, cl):
    model = None
    if args.backbone == 'mobilefacenet':
        model = MobileFacenet(cl)
    else:
        print(args.backbone, ' is not available!')
    return model


# gpu setting
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    args.use_gpu = True
    # model
    args.backbone = "mobilefacenet"
    # 加载预训练模型
    args.pretrained= True
    args.pretrained_model_path = "../pretrained_model/mobilefacenet/checkpoint/12-23_22-30/model/99_0.6626.pth"
    args.train_mode = "finetune"
    args.k = 0.8
    # feature dim
    args.feature_dim = 256
    # training parameters
    args.train_batch_size = 64
    args.test_batch_size = 32
    args.lr = 1e-3
    args.max_epoch = 200
    args.num_workers = 4
    # args.lfw_root = "../dataset/lfw-align-112x112"
    # args.lfw_train_list
    # optimizer
    args.optimizer = 'sgd' # ['sgd', 'adam', 'adabound']
    args.momentum = 0.9
    args.weight_decay = 5e-4
    # criterion
    args.criterion = 'FocalLoss' # ['CrossEntropyLoss', 'FocalLoss']
    args.use_center_loss = False
    args.weight_center_loss = 0.05
    # metric
    args.metric = 'arc_margin'  # ['arc_margin', 'add_margin', 'sphere']
    args.m = 0.5
    args.s = 32
    args.easy_margin = False
    # Scheduler
    args.lr_scheduler = 'MultiStep' # ['MultiStep', 'Step', 'ReduceLRonPlateau', 'CosineAnnealingLR']
    args.lr_step = 10 # Step
    args.gamma = 0.5 # Step, MultiStep
    args.milestones = [8, 16, 24, 32] # MultiStep
    args.factor = 0.1 # ReduceLRonPlateau
    args.patience = 5 # ReduceLRonPlateau
    # img
    args.img_mode = 'RGB' # ['RGB', 'L']
    args.input_shape = (112, 112)
    if args.img_mode == 'RGB':
        args.mean = [0.5, 0.5, 0.5]
        args.std = [0.5, 0.5, 0.5]

    # **************************   dataset      *******************************
    args.train_dataset = 'experiment'
    args.num_classes = 5754
    train_root = args.lfw_root
    train_list = args.lfw_train_list

    # *********************   checkpoint save path   **************************
    datetime = time.strftime('%m-%d_%H-%M')
    print('training start datetime:', datetime)
    args.save_path = os.path.join(args.save_path, 'checkpoint' , datetime)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for dir in ['model', 'metric', 'config']:
        dir_path = os.path.join(args.save_path, dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    # model
    cl, ll = get_layers("subnet")
    model = get_model(args, cl)
    if args.pretrained == True:
        model.load_state_dict(torch.load(args.pretrained_model_path), False)
        print('resume training: loaded pretrained model successfully!')
    prepare_model(model, args.train_mode, args.k)

    model.to(device)

    # models.metrics: arcface, cosine, sphere
    if args.metric == 'add_margin':
        metric_fc = AddMarginProduct(args.feature_dim, args.num_classes, s=30, m=0.35)
    elif args.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(args.feature_dim, args.num_classes, s=args.s, m=args.m, easy_margin=args.easy_margin) # s=60, m=0.5
    elif args.metric == 'sphere':
        metric_fc = SphereProduct(args.feature_dim, args.num_classes, m=4)
    else:
        metric_fc = nn.Linear(args.feature_dim, args.num_classes)
    metric_fc.to(device)

    # parallel training
    if len(args.gpu_id.split(',')) > 1:
        model = torch.nn.DataParallel(model)
        metric_fc = torch.nn.DataParallel(metric_fc)
        args.parallel = True

    # criterion
    criterion = None
    if args.criterion == 'FocalLoss':
        criterion = FocalLoss(gamma=2)
    elif args.criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = None
    optimizer_metric = None
    if args.optimizer == 'sgd':

        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=args.lr,  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        optimizer_metric = torch.optim.SGD([{'params': metric_fc.parameters(),'lr': 100*args.lr if args.pretrained else args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)

    # lr scheduler
    scheduler = None
    if args.lr_scheduler == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    elif args.lr_scheduler == 'ReduceLRonPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=True)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)
    elif args.lr_scheduler == 'MultiStep':
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_metric, milestones=args.milestones, gamma=args.gamma)

    # train dataset: webface or competition face
    train_dataset = Dataset(train_root, train_list, phase='train', input_shape=args.input_shape, img_mode=args.img_mode)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)

    iterations = 0
    for epoch in range(args.max_epoch):
        # training
        model.train()
        tbar = tqdm(train_loader, ncols=100)
        loss_sum = 0
        acc_sum = 0
        for ii, data in enumerate(tbar):
            iterations += 1
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input) # 512-d feature
            output = metric_fc(feature, label) # i-sample belongs each class probability
            loss = criterion(output, label) # loss

            optimizer.zero_grad() # clean gradient buffer
            optimizer_metric.zero_grad()
            loss.backward() # compute gradient
            optimizer.step() # update weight
            optimizer_metric.step()

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int)) # less meaning
            loss_sum += loss.item()
            acc_sum += acc
            tbar.set_description('epoch:%d, loss:%.4f, acc:%.4f' % (epoch, loss_sum/(ii+1), acc_sum/(ii+1)))
            # lfw_dataset validation
        if (epoch+1) % 10 == 0:
            model.eval()
            lfw_acc, lfw_th, t = lfw_test(model, device)
            print('lfw_accuracy:%.4f, threshold:%.4f, total_time:%ds' % (lfw_acc, lfw_th, int(t)), '\n')
            model.train()
        # save model and checkpoint
        save_model(model, metric_fc, epoch, round(acc_sum/(ii+1), 4))

        # update lr
        if args.lr_scheduler == 'Step' or args.lr_scheduler == 'CosineAnnealingLR' or args.lr_scheduler == 'MultiStep':
            scheduler.step()
        elif args.lr_scheduler == 'ReduceLRonPlateau':
            scheduler.step(loss_sum/(ii+1))