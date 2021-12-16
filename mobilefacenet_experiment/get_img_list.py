import os
from pathlib import Path

f = open('../dataset/dataset_list/lfw_test_list.txt', 'w')
root = Path('E:\server\competition\multi_race_face_recognition\public_dataset\lfw-align-112x112')
idx = 0
for identity_name in os.listdir(root):
    identity_path = root / identity_name
    for img_name in os.listdir(identity_path):
        img_path = identity_name + '/' + img_name
        f.write(img_path +'\n')
    idx += 1
f.close()