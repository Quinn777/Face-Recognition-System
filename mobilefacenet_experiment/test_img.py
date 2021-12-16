'''
    Get the cosine similarity of two pictures:
    Generally, if the similarity is bigger than the set thresh, we think the two images belong to the same person.
'''
import cv2
import os
import time
from tqdm import tqdm
import numpy as np

from PIL import Image
from mtcnn.detector import detect_faces
from mtcnn.models import PNet, RNet, ONet

# mobilefacenet
from backbone.mobilefacenet import MobileFacenet

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# device
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# LOAD MODELS
# mtcnn
pnet = PNet().to(device)
rnet = RNet().to(device)
onet = ONet().to(device)

# mobilefacenet
model = MobileFacenet()
test_model_path = '../pretrained_model/mobilefacenet/mobilefacenet.pth'
model.load_state_dict(torch.load(test_model_path))
model.to(device)
model.eval()


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def show_bboxes(img, bounding_boxes):
    H, W = img.shape[0], img.shape[1]
    H_max = 0
    box = None

    for BOX in bounding_boxes:
        h2_h1 = int(BOX[3]) - int(BOX[1])
        if h2_h1 > H_max:
            box = BOX
            H_max = h2_h1

    h1, h2, w1, w2 = int(box[1]), int(box[3]), int(box[0]), int(box[2])

    if h2 - h1 < 60:
        print('face too small\n')
        return img

    # tune the h
    h1 = int(h1 - (h2 - h1) / 12)
    # check the boundary
    if h1 < 0:
        h1 = 0
    height = h2 - h1

    # tune the w
    w_middle = int((w2 + w1) / 2)
    w1 = int(w_middle - height / 2)
    if w1 < 0:
        w1 = 0
    elif w1 + height >= W:
        w1 = W - height - 1
    w2 = w1 + height

    face_img = img[h1:h2, w1:w2, :]
    face_img = cv2.resize(face_img, (112, 112))
    cv2.imshow('face', cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(2000)
    return face_img

if __name__ == '__main__':
    # put two imgs in the img folder
    img_list = ['./img/test.jpg', './img/2.jpg']
    features = dict()
    for idx, img_path in enumerate(img_list):
        print('Idx:',idx, '  Img_path:',img_path)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(img_path, ' open img failed \n')
            continue
        start = time.time()
        try:
            bounding_boxes, landmarks = detec t_faces(img, pnet=pnet, rnet=rnet, onet=onet ,device=device)
        except:
            print(img_path, ' no img \n')
            continue
        end = time.time()
        img = np.asarray(img)
        face_img = show_bboxes(img, bounding_boxes)
        face_img = torch.from_numpy(face_img).float().permute(2, 0, 1).unsqueeze(0)
        face_img = face_img.to(device)
        with torch.no_grad():
            feature = model(face_img).cpu().squeeze().numpy()
            features[idx] = feature
    print('Similarity: ', cosin_metric(features[1], features[0]))



