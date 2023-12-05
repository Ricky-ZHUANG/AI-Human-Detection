import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
from torch import nn
import cv2
from torch.autograd import Variable

gpu = True
device = torch.device("cuda")
bsz = 32

from utils.FastF import get_magnitude_spectrum
from utils.noise_map import get_noise_map

#-----------------------------prepare data-------------------------------------#
class FaceDataset(Dataset):
    def __init__(self, real_root, fake_root):
        self.real_root = real_root
        images_path_real = Path(real_root)
        images_list_real = list(images_path_real.glob('*.jpg'))
        images_list_str_real = [ str(x) for x in images_list_real ]
        true_label = [1 for _ in images_list_real]

        images_path_fake = Path(fake_root)
        images_list_fake = list(images_path_fake.glob('*.jpg'))
        images_list_str_fake = [ str(x) for x in images_list_fake ]
        false_label = [0 for _ in images_list_fake]

        self.images = images_list_str_real+images_list_str_fake
        self.labels = true_label+false_label
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        oroginal = cv2.imread(image_path)
        oroginal = torch.Tensor(oroginal).permute(2, 0, 1)
        noise = get_noise_map(image_path)
        noise = torch.Tensor(noise).permute(2, 0, 1)
        magnitude = get_magnitude_spectrum(image_path)
        magnitude = torch.Tensor(magnitude).unsqueeze(0)#.permute(2, 0, 1)
        label = self.labels[idx]

        sample = {
            "original" : oroginal,
            "noise" : noise,
            "magnitude" : magnitude,
            "label" : label
        }
        return sample

class SingleDataset(Dataset):
    def __init__(self, root):
        # self.real_root = real_root
        # images_path_real = Path(real_root)
        # images_list_real = list(images_path_real.glob('*.jpg'))
        # images_list_str_real = [ str(x) for x in images_list_real ]
        # true_label = [1 for _ in images_list_real]

        images_path = Path(root)
        images_list = list(images_path.glob('*.jpg'))
        images_list_str = [ str(x) for x in images_list ]
        false_label = [0 for _ in images_list]

        self.images = images_list_str
        self.labels = false_label
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        oroginal = cv2.imread(image_path)
        oroginal = torch.Tensor(oroginal).permute(2, 0, 1)
        noise = get_noise_map(image_path)
        noise = torch.Tensor(noise).permute(2, 0, 1)
        magnitude = get_magnitude_spectrum(image_path)
        magnitude = torch.Tensor(magnitude).unsqueeze(0)#.permute(2, 0, 1)
        label = self.labels[idx]

        sample = {
            "original" : oroginal,
            "noise" : noise,
            "magnitude" : magnitude,
            "label" : label
        }
        return sample

# real_root_train = 'face dataset/train/real'
# fake_root_train = 'face dataset/train/fake'
real_root_test = 'face dataset/test/real'
fake_root_test = 'face dataset/test/fake'

test_easy = ''
test_hard = ''
test_mid = ''


# train = FaceDataset(real_root=real_root_train,fake_root=fake_root_train)
# train_loader = DataLoader(train, batch_size=bsz, shuffle=True)
test = FaceDataset(real_root=real_root_test,fake_root=fake_root_test)
test_loader = DataLoader(test, batch_size=bsz, shuffle=True)

# easy = FakeDataset(fake_root=test_easy)
# easy_loader = DataLoader(easy, batch_size=bsz, shuffle=True)
# hard = FakeDataset(fake_root=test_hard)
# hard_loader = DataLoader(hard, batch_size=bsz, shuffle=True)
# mid = FakeDataset(fake_root=test_easy)
# mid_loader = DataLoader(mid, batch_size=bsz, shuffle=True)
#-----------------------------prepare data-------------------------------------#
print('prepare data end')
print('-'*89)
#-----------------------------define model-------------------------------------#
####################baseline#######################
# from models.ViT import VisionTransformerEncoder
# # from models.VGGNet import VGGEncoder
# from models.ResNet import ResNetEncoder
# from models.MLPClassifier import MLPClassifier
# # from models.CNN import CNNEncoder

# class BaseLineModel(nn.Module):
#     def __init__(self, encoder, classifier):
#         super(BaseLineModel, self).__init__()
#         self.encoder = encoder
#         self.classifier = classifier

#     def forward(self, img):
#         img_emb = self.encoder(img)
#         pred = self.classifier(img_emb)
        
#         return pred

# # vit_encoder = VisionTransformerEncoder()
# # cnn_encoder = CNNEncoder()
# resnet_encoder = ResNetEncoder()
# mlp_classifier = MLPClassifier(fc_in_dim=1500,fc_hid_dim=[512, 512])

# model = BaseLineModel(encoder=resnet_encoder, classifier=mlp_classifier)
# if gpu:
#     model = model.to(device)
# bce_loss = torch.nn.BCELoss()
####################baseline#######################

####################facenet#######################
# from models.CNN import CNNEncoder
from models.ResNet import ResNetEncoder
# from models.VGGNet import VGGEncoder
from models.ViT import VisionTransformerEncoder
from models.MLPClassifier import MLPClassifier

from models.FaceNet import MultiFacePredictor
from models.FaceNet import MergingModel
from models.FaceNet import FourierEncoder

fourier_encoder = FourierEncoder()
noise_encoder = ResNetEncoder()
face_encoder = VisionTransformerEncoder()
predictor = MLPClassifier()
merging_layer = MergingModel()

model = MultiFacePredictor(fourier_encoder=fourier_encoder, noise_encoder=noise_encoder, face_encoder=face_encoder, predictor=predictor, merging_layer=merging_layer)
if gpu:
    model = model.to(device)
bce_loss = torch.nn.BCELoss()
####################facenet#######################


# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Wrap the model with DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)

model.load_state_dict(torch.load('FaceNetBaseline.pt'))
#-----------------------------define model-------------------------------------#
print('define model end')
print('-'*89)
#-----------------------------test model auroc general-------------------------------------#
# from torcheval.metrics import BinaryAUROC
# from torcheval.metrics import BinaryAUPRC
# from torcheval.metrics.functional import binary_f1_score as f1
# auroc = BinaryAUROC()
# auprc = BinaryAUPRC()
# def evaluate(data_loader, loss_metric):
#     model.eval()
#     total_loss = 0
#     total_f1 = 0
#     correct = 0    
#     total = 0 

#     with torch.no_grad():
#         for iteration, sample in enumerate(data_loader):
#             original = Variable(sample['original'])
#             noise = Variable(sample['noise'])
#             magnitude = Variable(sample['magnitude'])

#             label_true = Variable(sample['label'].float())

#             if gpu:
#                 noise=noise.to(device)
#                 original=original.to(device)
#                 magnitude=magnitude.to(device)
#                 label_true=label_true.to(device)    

#             # label = model(original).view(-1)
#             label = model(fourier=magnitude, noise=noise, face=original).view(-1)
#             loss_metric.update(label,label_true)
#             loss = loss_metric.compute()
#             f1_socre = f1(label, label_true)
#             total_loss += loss.data
#             total_f1 += f1_socre
#             # total += label_true.size(0)
#             # correct += (label == label_true).sum().item()  
#     print('Test score: ', total_loss.item()/(iteration+1)) 
#     print('Test F1: ', total_f1.item()/(iteration+1))
#     return total_loss.item()/(iteration+1)

# evaluate(test_loader, loss_metric=auroc)
# evaluate(test_loader, loss_metric=f1)
#-----------------------------test model auroc general-------------------------------------#

#-----------------------------test model auprc/f1 easy/middle/hard-------------------------------------#
# test_easy = 'face dataset/test/fake_easy'
# test_hard = 'face dataset/test/fake_hard'
# test_middle = 'face dataset/test/fake_middle'

# test = FaceDataset(real_root=real_root_test,fake_root=test_hard)
# test_loader = DataLoader(test, batch_size=bsz, shuffle=True)

# evaluate(test_loader, loss_metric=auprc)
#-----------------------------test model auprc/f1 easy/middle/hard-------------------------------------#

#-----------------------------Single(true/fake) testing-------------------------------------#
def evaluate(data_loader):
    model.eval()
    labels = np.array([])

    with torch.no_grad():
        for iteration, sample in enumerate(data_loader):
            original = Variable(sample['original'])
            noise = Variable(sample['noise'])
            magnitude = Variable(sample['magnitude'])

            label_true = Variable(sample['label'].float())

            if gpu:
                noise=noise.to(device)
                original=original.to(device)
                magnitude=magnitude.to(device)
                label_true=label_true.to(device)    

            # label = model(original).view(-1)
            label = model(fourier=magnitude, noise=noise, face=original).view(-1)
            labels = np.append(labels, label.cpu().data.numpy())
    return labels

test = SingleDataset(root=fake_root_test)
test_loader = DataLoader(test, batch_size=bsz, shuffle=True)

result = evaluate(test_loader).flatten()
print(np.array2string(result, separator=","))
#-----------------------------Single(true/fake) testing-------------------------------------#
