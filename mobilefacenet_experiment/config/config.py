import argparse

parse = argparse.ArgumentParser(description='arcface to do face recognition')
# lfw dataset for test
parse.add_argument('--lfw_root', type=str, default='../dataset/lfw-align-112x112')
# parse.add_argument('--lfw_root', type=str, default='E:\server\competition\multi_race_face_recognition\public_dataset/lfw-align-112x112')
parse.add_argument('--lfw_train_list', type=str, default='../dataset/dataset_list/lfw_train_list.txt')
parse.add_argument('--lfw_test_list', type=str, default='../dataset/dataset_list/lfw_test_list.txt')
parse.add_argument('--lfw_test_pair', type=str, default='../dataset/dataset_list/lfw_test_pair.txt')

# save path: model.pth, checkpoint.pth, config.json
parse.add_argument('--pretrained', type=bool, default=False)
parse.add_argument('--pretrained_model_path', type=str)
parse.add_argument('--save_path', type=str, default='../pretrained_model/mobilefacenet')

# super parameters
parse.add_argument('--use_gpu', type=bool, default=False)
parse.add_argument('--gpu_id', type=str, default='0')
parse.add_argument('--parallel', type=bool, default=False)
parse.add_argument('--input_shape', type=list, default=[112,112])
parse.add_argument('--feature_dim', type=int , default=256)
parse.add_argument('--train_batch_size', type=int, default=128)
parse.add_argument('--test_batch_size', type=int, default=1024)
parse.add_argument('--num_workers', type=int, default=4)

parse.add_argument('--max_epoch', type=int, default=50)
parse.add_argument('--lr', type=float, default=1e-1)
parse.add_argument('--optimizer', type=str, default='sgd')
parse.add_argument('--momentum', type=float, default=0.9)
parse.add_argument('--weight_decay', type=float, default=5e-4)

parse.add_argument('--lr_scheduler', type=str, default='Step')
parse.add_argument('--lr_step', type=int, default=10)
parse.add_argument('--gamma', type=float, default=0.1)
parse.add_argument('--patience', type=int, default=5)
parse.add_argument('--factor', type=float, default=0.2)

parse.add_argument('--img_mode', type=str, default='RGB')
parse.add_argument('--mean', type=list)
parse.add_argument('--std', type=list)


# training scheduler
parse.add_argument('--num_classes', type=int)
parse.add_argument('--backbone', type=str, default='mobilefacenet')
parse.add_argument('--dropout_rate', type=float, default=0.5)
parse.add_argument('--metric', type=str, default='arc_margin')
parse.add_argument('--m', type=float, default=0.5)
parse.add_argument('--s', type=int, default=64)
parse.add_argument('--use_se', type=bool, default=True)
parse.add_argument('--easy_margin', type=bool, default=False)

# parse.add_argument('--classify', type=str, default='softmax')
parse.add_argument('--criterion', type=str, default='CrossEntropyLoss')
parse.add_argument('--use_center_loss', type=bool, default=False)
parse.add_argument('--weight_center_loss', type=float, default=0.01)

# submission result
args = parse.parse_args()