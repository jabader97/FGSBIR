from torch.autograd import Variable
import torch.nn as nn
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network, VGG_with_Attention_Network
from torch import optim
import torch
import time
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(hp.backbone_name + '_Network(hp)')
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.optimizer = optim.Adam(self.sample_train_params, hp.learning_rate)
        torch.manual_seed(hp.seed)
        self.hp = hp

    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()

        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        negative_feature = self.sample_embedding_network(batch['negative_img'].to(device))
        sample_feature = self.sample_embedding_network(batch['sketch_img'].to(device))

        loss = self.loss(sample_feature, positive_feature, negative_feature)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def test_forward(self, batch):            #  this is being called only during evaluation
        sketch_feature = self.sample_embedding_network(batch['sketch_img'].to(device))
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        return sketch_feature.cpu(), positive_feature.cpu()



