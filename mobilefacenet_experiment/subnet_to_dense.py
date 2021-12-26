from backbone.mobilefacenet import MobileFacenet
from backbone.utils import get_layers, prepare_model, subnet_to_dense, current_model_pruned_fraction
import torch
from lfw_test import lfw_test

cl, ll = get_layers("dense")
model = MobileFacenet(cl)
test_model_path = '../pretrained_model/mobilefacenet/checkpoint/12-23_16-41/model/188_0.9649.pth'
# model.load_state_dict(torch.load(test_model_path, map_location=torch.device('cpu')))

sub_state_dict = torch.load(test_model_path)
dense_state_dict = subnet_to_dense(sub_state_dict, 0.5)

# torch.save(dense_state_dict, '../pretrained_model/mobilefacenet/checkpoint/12-23_16-41/model/188_0.9649-dense.pth')

model.load_state_dict(dense_state_dict, False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
lfw_acc, lfw_th, t = lfw_test(model,device)



prepare_model(model, "test", 1.0)
# prune_rate = current_model_pruned_fraction(model, sub_state_dict)
print()