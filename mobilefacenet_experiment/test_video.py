import cv2
import os
import numpy as np
from PIL import Image
from mobilefacenet_experiment.mtcnn.tools.test_detect import MtcnnDetector
from backbone.mobilefacenet import MobileFacenet
from backbone.utils import get_layers, prepare_model, subnet_to_dense
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model,
cl, ll = get_layers("dense")
model = MobileFacenet(cl)
# 加载模型（未剪枝）
test_model_path = '../pretrained_model/mobilefacenet/checkpoint/12-22_19-53/model/95_0.9876.pth'
weight = torch.load(test_model_path, map_location=torch.device('cpu'))
model.load_state_dict(weight, False)
model.eval()
model.to(device)


def cosin_metric(x1, x2):
    """
    余弦相似度
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def get_predict_name(feature):
    """
    将测试图片与标签图片对比，获取相似度最高的标签作为输出
    """
    max_sim = -10
    pre_name = None
    for key in list(features_dict.keys()):
        sim = cosin_metric(feature, features_dict[key])

        if sim > max_sim:
            max_sim = sim
            pre_name = key
    if max_sim < 0.5:
        pre_name = 'unknown'
    if "_" in pre_name:
        pre_name = pre_name.split("_")[0]
    print(pre_name, max_sim)
    return pre_name+": "+str(max_sim)


def crop_face(img, bounding_boxes):
    """
    从原始图片中裁剪出人脸
    """
    H, W = img.shape[0], img.shape[1]
    H_max = 0
    box = None

    for BOX in bounding_boxes:
        h2_h1 = int(BOX[3]) - int(BOX[1])
        if h2_h1 > H_max:
            box = BOX
            H_max = h2_h1

    h1, h2, w1, w2 = int(box[1]), int(box[3]), int(box[0]), int(box[2])

    # 调整高度
    h1 = int(h1 - (h2 - h1) / 12)
    # check the boundary
    if h1 < 0:
        h1 = 0
    height = h2 - h1

    # 调整宽度
    w_middle = int((w2 + w1) / 2)
    w1 = int(w_middle - height / 2)
    if w1 < 0:
        w1 = 0
    elif w1 + height >= W:
        w1 = W - height - 1
    w2 = w1 + height
    # 截取图像
    face_img = img[h1:h2, w1:w2, :]
    face_img = cv2.resize(face_img, (112, 112))
    return face_img, box


def draw_images(img, bboxs, landmarks):
    """
    在图片上绘制人脸框及特征点
    """
    face_num = bboxs.shape[0]
    for i in range(face_num):
        cv2.rectangle(img, (int(bboxs[i, 0]), int(bboxs[i, 1])), (int(bboxs[i, 2]), int(bboxs[i, 3])), (0, 255, 0), 2)
    for p in landmarks:
        for i in range(5):
            cv2.circle(img, (int(p[2 * i]), int(p[2 * i + 1])), 2, (0, 0, 255), -1)
    return img


def get_face_feature(model, face_img):
    """
    提取人脸特征
    """
    face_img = face_img.astype(np.float32)
    face_img /= 255.0
    face_img -= (0.5, 0.5, 0.5)
    face_img /= (0.5, 0.5, 0.5)
    face_img = torch.from_numpy(face_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(face_img)
        feature = feature.squeeze().cpu().numpy()
        return feature


def get_your_faceImg_feature(name, model, features_dict, mtcnn_detector):
    """
    将带有人名标签的图片提取特征并保存，以便于后续对比
    :param name: 对应的标签人名
    :param model: 特征提取模型
    :param features_dict: 作为标签的特征
    :param mtcnn_detector: 检测模型
    :return: 加入了新标签的特征字典
    """
    faceImg_path = os.path.join(r'.\img', name)
    num_img_useful = 0
    for idx, img_name in enumerate(os.listdir(faceImg_path)):
        # detect face in your img
        img = Image.open(faceImg_path + '/' + img_name).convert('RGB')
        try:
            img = np.array(img)
            RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bounding_boxes, landmarks = mtcnn_detector.detect_face(RGB_image)
        except:
            continue
        if len(bounding_boxes) == 0:
            continue
        # crop the face from the img (112, 112)
        face_img, box = crop_face(np.array(img), bounding_boxes)
        feature = get_face_feature(model, face_img)
        features_dict[name+'_'+str(idx)] = feature
        num_img_useful += 1
    print('----------Successfully collected your face img for %ds ----------' % num_img_useful)
    return features_dict


if __name__ == '__main__':
    # 人脸检测模型
    mtcnn_detector = MtcnnDetector(min_face_size=24, use_cuda=False)
    features_dict = {}
    # 使用自己的照片测试
    test_your_video = True
    if test_your_video :
        dirs = r"./img"
        for file_name in os.listdir(dirs):
            print(file_name)
            features_dict = get_your_faceImg_feature(file_name, model, features_dict, mtcnn_detector)

    print('Features length:', len(features_dict))
    # 测试视频路径
    video_src = './video/WJQ_2.mp4'

    # 逐帧进行人脸识别过程
    capture = cv2.VideoCapture(video_src)
    if not capture.isOpened():
        print('Camera is not opened or the video src doesnt exit!')
    else:
        idx_frame = 0
        while True:
            # 取帧
            ret, frame = capture.read()
            if not ret:
                break
            idx_frame += 1
            # 每2帧预测一次
            if idx_frame % 2 != 0:
                continue
            idx_frame = 0
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 检测图像中是否存在人脸
            try:
                bboxs, landmarks = mtcnn_detector.detect_face(rgb_img)
            except:
                continue
            if len(bboxs) ==0:
                continue
            # 从原始图像中裁剪出人脸
            face_img, box = crop_face(frame, bboxs)
            cv2.imshow('face', face_img)
            # 获取人脸特征
            feature = get_face_feature(model, face_img)
            # 将测试帧中的特征信息与特征库比对，获取预测结果
            pre_name = get_predict_name(feature)
            # 可视化结果
            cv2.putText(frame, pre_name, (20, 40), 0, 1, (0, 255, 0), 2)
            num_face = bboxs.shape[0]
            for i in range(num_face):
                cv2.rectangle(frame, (int(bboxs[i, 0]), int(bboxs[i, 1])), (int(bboxs[i, 2]), int(bboxs[i, 3])),
                              (0, 255, 0), 2)
            for p in landmarks:
                for i in range(5):
                    cv2.circle(frame, (int(p[2 * i]), int(p[2 * i + 1])), 2, (0, 0, 255), -1)
            cv2.imshow('Prediction', frame)
            cv2.waitKey(1)
    capture.release()
    print("----------Video Recognition Finished----------")
