import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFacePredictor(nn.Module):
    def __init__(self, fourier_encoder, noise_encoder, face_encoder, predictor, merging_layer):
        super(MultiFacePredictor, self).__init__()
        self.fourier_encoder = fourier_encoder
        self.noise_encoder = noise_encoder
        self.face_encoder = face_encoder
        self.predictor = predictor
        self.merging_layer =merging_layer
    
    def forward(self, fourier, noise, face):
        fourier_emb = self.fourier_encoder(fourier)
        noise_emb = self.noise_encoder(noise)
        face_emb = self.face_encoder(face)

        multi_model_emb = self.merging_layer(fourier_emb, noise_emb, face_emb)
        pred = self.predictor(multi_model_emb)

        return pred
    
class MergingModel(nn.Module):
    def __init__(self, fc_in_dim=1500*3, fc_hid_dim=[512, 512], embed_dim=1500, dropout=0.5):
        super(MergingModel, self).__init__()
        self.fc_hid_dim = fc_hid_dim
        self.fc = nn.Linear(fc_in_dim, self.fc_hid_dim[0])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.classifier = nn.ModuleList()

        for input_size, output_size in zip(self.fc_hid_dim, self.fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                            #   nn.BatchNorm1d(output_size),
                            #   self.act,
                              nn.Dropout(p=self.dropout)
                             )
            )
        self.fc2 = nn.Linear(self.fc_hid_dim[-1], embed_dim)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)

    def forward(self, info1, info2, info3):
        x = torch.cat((info1, info2, info3),1)
        x = F.relu(self.fc(x))
        for fc in self.classifier:
            x = F.relu(fc(x))
        x = self.fc2(x)
        return x

class FourierEncoder(nn.Module):
    def __init__(self):
        super(FourierEncoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc = nn.Linear(4096, 1500)  # Fully connected layer to get a vector of size 1500x1
        self.dropout=0.2
   

        channels = [1, 16, 32, 64, 128, 256, 512, 1024,1024]

        for input_dim, output_dim in zip(channels, channels[1:]):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1), 
                    # nn.BatchNorm2d(output_dim), 
                    # nn.MaxPool2d(kernel_size=2, stride=2),
                    # nn.ReLU(),
                    )
            )

        for layer in self.conv_layers:
            nn.init.xavier_uniform_(layer[0].weight)
 
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.relu(self.mp(conv_layer(x)))
        
        x = self.flatten(x)
        # x = x.view(-1)
        x = self.fc(x)
        
        return x
    

