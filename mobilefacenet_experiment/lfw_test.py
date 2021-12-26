'''
    Use lfw dataset to test the performance of the face recognition model.
'''
import os
import time
import numpy as np
from tqdm import tqdm

import torch
from dataset import Dataset
from torch.utils.data import DataLoader
from config.config import args

from backbone.mobilefacenet import MobileFacenet

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)
        # f.write(str(sim) + '\n')
        sims.append(sim)
        labels.append(label)
    acc, th = cal_accuracy(sims, labels)
    return acc, th

def get_features(model, device):
    dataset = Dataset(root=args.lfw_root,
                      dataset_list=args.lfw_test_list,
                      phase='test',
                      input_shape=args.input_shape,
                      img_mode = args.img_mode)
    test_dataloader = DataLoader(dataset, batch_size=32)
    tbar = tqdm(test_dataloader)
    features_dict = {}
    with torch.no_grad():
        for idx, (img, name) in enumerate(tbar):
            img = img.to(device)
            features = model(img)
            features = features.detach().cpu().numpy()
            for face, feature in zip(name, features):
                features_dict[face] = feature
        return features_dict

def lfw_test(model, device):
    s = time.time()
    features_dict = get_features(model, device)
    t = time.time() - s
    acc, th = test_performance(features_dict, args.lfw_test_pair)
    return acc, th, t

def get_model():
    model = MobileFacenet()
    return model


if __name__ == '__main__':
    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model path
    args.input_shape = [112, 112]
    args.test_batch_size = 512
    args.test_model_path = '../pretrained_model/mobilefacenet/mobilefacenet.pth'

    # loading model
    model = get_model()
    model.load_state_dict(torch.load(args.test_model_path))
    model.eval()
    model.to(device)
    print('finish loading model')

    acc, th, t = lfw_test(model, device)
    print('lfw_accuracy:%.4f, threshold:%.4f, total_time:%ds' % (acc, th, int(t)), '\n')



