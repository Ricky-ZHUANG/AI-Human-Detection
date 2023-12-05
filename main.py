import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import matplotlib
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torch import nn

from utils.FastF import get_magnitude_spectrum
from utils.noise_map import get_noise_map

bsz = 64
log_interval = 3
gpu = True
device = torch.device("cuda")
real_root_train = 'face dataset/train/real'
fake_root_train = 'face dataset/train/fake'
real_root_test = 'face dataset/valid/real'
fake_root_test = 'face dataset/valid/fake'

#-----------------------------prepare data-------------------------------------#
class FaceDataset(Dataset):
    def __init__(self, real_root, fake_root):
        self.real_root = real_root
        images_path_real = Path(real_root)
        images_list_real = list(images_path_real.glob('*.jpg'))
        images_list_str_real = [ str(x) for x in images_list_real if not str(x).startswith(real_root+'/._')]
        true_label = [1 for _ in images_list_real]

        images_path_fake = Path(fake_root)
        images_list_fake = list(images_path_fake.glob('*.jpg'))
        images_list_str_fake = [ str(x) for x in images_list_fake if not str(x).startswith(fake_root+'/._')]
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

train = FaceDataset(real_root=real_root_train,fake_root=fake_root_train)
train_loader = DataLoader(train, batch_size=bsz, shuffle=True)
test = FaceDataset(real_root=real_root_test,fake_root=fake_root_test)
test_loader = DataLoader(test, batch_size=bsz, shuffle=True)
#-----------------------------prepare data-------------------------------------#
print('prepare data end')
print('-'*89)
#-----------------------------define model-------------------------------------#
# from models.CNN import CNNEncoder
from models.ResNet import ResNetEncoder
# from models.VGGNet import VGGEncoder
from models.ViT import VisionTransformerEncoder
from models.MLPClassifier import MLPClassifier

from models.FaceNet import MultiFacePredictor
from models.FaceNet import MergingModel
from models.FaceNet import FourierEncoder

# ------------------------baseline----------------------
class BaseLineModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(BaseLineModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, img):
        img_emb = self.encoder(img)
        pred = self.classifier(img_emb)
        
        return pred

vit_encoder = VisionTransformerEncoder()
# vgg_encoder = VGGEncoder()
# resnet_encoder = ResNetEncoder()
mlp_classifier = MLPClassifier(fc_in_dim=1500,fc_hid_dim=[512, 512])

model = BaseLineModel(encoder=vit_encoder, classifier=mlp_classifier)
if gpu:
    model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
bce_loss = torch.nn.BCELoss()
# ------------------------baseline----------------------

# ------------------------face net----------------------
# fourier_encoder = FourierEncoder()
# noise_encoder = ResNetEncoder()
# face_encoder = VisionTransformerEncoder()
# predictor = MLPClassifier()
# merging_layer = MergingModel()

# model = MultiFacePredictor(fourier_encoder=fourier_encoder, noise_encoder=noise_encoder, face_encoder=face_encoder, predictor=predictor, merging_layer=merging_layer)
# if gpu:
#     model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# bce_loss = torch.nn.BCELoss()
# ------------------------face net----------------------

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Wrap the model with DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
#-----------------------------define model-------------------------------------#
print('define model end')
print('-'*89)
#-----------------------------training-------------------------------------#
import matplotlib.pyplot as plt
# ------------------------baseline----------------------
def training(data_loader):
    model.train()
    total_loss = 0
    epoch_loss = 0 #mean loss of this epoch

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

        optimizer.zero_grad()
        label = model(original).view(-1)
        loss = bce_loss(label,label_true)
        loss.backward()
        optimizer.step() 
        total_loss += loss.data
        epoch_loss += loss.data

        if iteration%log_interval == 0 and iteration > 0:
            # print('total loss: ',str(total_loss))
            cur_loss = total_loss.item() / log_interval
            print('| epoch {:3d} | batches {:5d} | loss {:8.5f}'.format(epoch,iteration,cur_loss))
            total_loss = 0
    training_loss.append(epoch_loss.item()/(iteration+1))
    print('Epoch BCE loss: ',epoch_loss.item()/(iteration+1))

def evaluate(data_loader):
    model.eval()
    total_loss = 0
    correct = 0    
    total = 0 

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

            label = model(original).view(-1)
            loss = bce_loss(label,label_true)
            total_loss += loss.data
            # total += label_true.size(0)
            # correct += (label == label_true).sum().item()  
    testing_loss.append(total_loss.item()/(iteration+1))
    print('Test BCE loss: ', total_loss.item()/(iteration+1)) 
    # print(f'{correct} corrects out of {total} samples')
    # print('Test accuracy: ', (correct/total)) 


epochs = 1
training_loss = []
testing_loss = []
try:
    for epoch in range(1, epochs+1):
        # epoch_start_time = time.time()
        training(train_loader)
        evaluate(test_loader)
        print('-'*89)
    plt.plot(training_loss, color='red', label='Training Loss')
    plt.plot(testing_loss, color='blue', label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Training and Testing Loss')
    # plt.savefig('ResNet_Result_Fig.png')
    plt.legend()
    plt.show()
except KeyboardInterrupt:
    print('-'*89)
    print('Existing from training early')   
# ------------------------baseline----------------------

# ------------------------face net----------------------
# def training(data_loader):
#     model.train()
#     total_loss = 0
#     epoch_loss = 0 #mean loss of this epoch

#     for iteration, sample in enumerate(data_loader):
#         original = Variable(sample['original'])
#         noise = Variable(sample['noise'])
#         magnitude = Variable(sample['magnitude'])

#         label_true = Variable(sample['label'].float())

#         if gpu:
#             noise=noise.to(device)
#             original=original.to(device)
#             magnitude=magnitude.to(device)
#             label_true=label_true.to(device)

#         optimizer.zero_grad()
#         label = model(fourier=magnitude, noise=noise, face=original).view(-1)
#         loss = bce_loss(label,label_true)
#         loss.backward()
#         optimizer.step() 
#         total_loss += loss.data
#         epoch_loss += loss.data

#         if iteration%log_interval == 0 and iteration > 0:
#             # print('total loss: ',str(total_loss))
#             cur_loss = total_loss.item() / log_interval
#             print('| epoch {:3d} | batches {:5d} | loss {:8.5f}'.format(epoch,iteration,cur_loss))
#             total_loss = 0
#     training_loss.append(epoch_loss.item()/(iteration+1))
#     print('Epoch BCE loss: ',epoch_loss.item()/(iteration+1))

# def evaluate(data_loader):
#     model.eval()
#     total_loss = 0
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

#             label = model(fourier=magnitude, noise=noise, face=original).view(-1)
#             loss = bce_loss(label,label_true)
#             total_loss += loss.data
#             # total += label_true.size(0)
#             # correct += (label == label_true).sum().item()  
#     testing_loss.append(total_loss.item()/(iteration+1))
#     print('Test BCE loss: ', total_loss.item()/(iteration+1)) 
#     return total_loss.item()/(iteration+1)
#     # print(f'{correct} corrects out of {total} samples')
#     # print('Test accuracy: ', (correct/total)) 

# epochs = 10
# training_loss = []
# testing_loss = []
# min_loss = 100
# try:
#     for epoch in range(1, epochs+1):
#         # epoch_start_time = time.time()
#         training(train_loader)
#         loss = evaluate(test_loader)
#         if (loss < min_loss):
#             min_loss = loss
#             torch.save(model.state_dict(), 'ResNetBaseline.pt')
#             print(f'save model with min bce loss: {min_loss}')
#         print('-'*89)
#     plt.plot(training_loss, color='red', label='Training Loss')
#     plt.plot(testing_loss, color='blue', label='Testing Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('BCE Loss')
#     plt.title('Training and Testing Loss')
#     plt.savefig('FaceNet_Result_Fig.png')
#     plt.legend()
#     plt.show()
# except KeyboardInterrupt:
#     print('-'*89)
#     print('Existing from training early')
# ------------------------face net----------------------
#-----------------------------training-------------------------------------#
torch.save(model.state_dict(), 'ViTNetBaseline.pt')
#-----------------------------test model-------------------------------------#
real_root_test = 'face dataset/test/real'
fake_root_test = 'face dataset/test/fake'
test = FaceDataset(real_root=real_root_test,fake_root=fake_root_test)
test_loader = DataLoader(test, batch_size=bsz, shuffle=True)

from torcheval.metrics import BinaryAUROC
from torcheval.metrics.functional import binary_f1_score as f1

auroc = BinaryAUROC()
def evaluate(data_loader, loss_metric):
    model.eval()
    total_loss = 0
    total_f1 = 0
    correct = 0    
    total = 0 

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

            label = model(original).view(-1)
            loss_metric.update(label,label_true)
            loss = loss_metric.compute()
            f1_score = f1(label,label_true)
            total_f1 += f1_score.data
            total_loss += loss.data
            # total += label_true.size(0)
            # correct += (label == label_true).sum().item()  
    print('Test loss: ', total_loss.item()/(iteration+1)) 
    print(f'f1: {total_f1.item()/(iteration+1)}')
    return total_loss.item()/(iteration+1)

evaluate(test_loader, loss_metric=auroc)