import os
import cv2
import json
from tqdm import tqdm
import numpy as np


import torch

from backbone.model import MobileFacenet

# model
def get_model(test_model_path):
    model = MobileFacenet()
    model.load_state_dict(torch.load(test_model_path))
    model.eval()
    model.to(device)
    return model

# get features dict and save
def get_features(model, root_list):
    features = dict()
    tbar = tqdm(total=7804)
    with torch.no_grad():
        for root in root_list: # lab and undergraduate
            for identity_name in os.listdir(root):
                img_dir = os.path.join(root, identity_name)
                for idx, img_name in enumerate(os.listdir(img_dir)):
                    img_path = os.path.join(img_dir, img_name)
                    face_img = cv2.imread(img_path)
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
                    face_img /= 255.0
                    face_img -= (0.5, 0.5, 0.5)
                    face_img /= (0.5, 0.5, 0.5)
                    face_img = torch.from_numpy(face_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    feature = model(face_img).cpu().squeeze().numpy()
                    features[identity_name+'_'+str(idx)] = feature.tolist()
                    tbar.update(1)
    print('Finish collecting the face features!')
    print('Num of features:', len(features))
    json.dump(features, open('../pretrained_model/mobilefacenet/features_dict/lab_undergraduate_visible_features.json', 'w'), indent=4)
    return features


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model_path = '../pretrained_model/mobilefacenet/mobilefacenet.pth'
    root_list = ['E:/face-dataset/real_img/lab/visible-light-align-112',
                 'E:/face-dataset/real_img/undergraduate/visible-light-align-112']
    model = get_model(test_model_path)
    get_features(model, root_list)