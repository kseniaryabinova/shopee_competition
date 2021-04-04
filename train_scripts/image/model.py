import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from torchsummary import summary

import timm


class ArcModule(nn.Module):
    def __init__(self, device, in_features, out_features, s=10, margin=0.5, is_amp=False):
        super().__init__()
        self.is_amp = is_amp
        self.device = device

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = torch.tensor(math.cos(math.pi - margin))
        self.mm = torch.tensor(math.sin(math.pi - margin) * margin)

    def forward(self, inputs, labels):
        if self.is_amp:
            with autocast():
                cos_th = F.linear(inputs, F.normalize(self.weight))
                cos_th = cos_th.clamp(-1, 1)
                sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
                cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
                cos_th_m = torch.where(cos_th.half() > self.th.half(), cos_th_m.half(), cos_th.half() - self.mm.half()).half()

                cond_v = cos_th - self.th
                cond = cond_v <= 0
                cos_th_m[cond] = (cos_th - self.mm)[cond]

                if labels.dim() == 1:
                    labels = labels.unsqueeze(-1)
                onehot = torch.zeros(cos_th.size()).to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)
                onehot.scatter_(1, labels, 1.0)
                outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
                outputs = outputs * self.s

        else:
            cos_th = F.linear(inputs, F.normalize(self.weight))
            cos_th = cos_th.clamp(-1, 1)
            sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
            cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
            # print(type(cos_th), type(self.th), type(cos_th_m), type(self.mm))
            cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

            cond_v = cos_th - self.th
            cond = cond_v <= 0
            cos_th_m[cond] = (cos_th - self.mm)[cond]

            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            onehot = torch.zeros(cos_th.size()).to(self.device)
            labels = labels.type(torch.LongTensor).to(self.device)
            onehot.scatter_(1, labels, 1.0)
            outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
            outputs = outputs * self.s

        return outputs


class EfficientNetArcFace(nn.Module):

    def __init__(self, channel_size, out_feature, device, dropout=0.5, backbone='tf_efficientnet_b0_ns',
                 pretrained=True, margin=0.5, is_amp=False):
        super(EfficientNetArcFace, self).__init__()
        self.is_amp = is_amp

        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.in_features = self.backbone.classifier.in_features
        self.backbone.global_pool = nn.Identity()
        self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.channel_size = channel_size
        self.out_feature = out_feature
        self.margin = ArcModule(device, in_features=self.channel_size, out_features=self.out_feature, margin=margin,
                                is_amp=self.is_amp)
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features, self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):
        if self.is_amp:
            with autocast():
                features = self.backbone(x)
                features = self.pooling(features)

                features = self.bn1(features)
                features = self.dropout(features)
                features = features.view(features.size(0), -1)
                features = self.fc1(features)
                features = self.bn2(features)
                features = F.normalize(features)
                if labels is not None:
                    return self.margin(features, labels)
                return features

        else:
            features = self.backbone(x)
            features = self.pooling(features)

            features = self.bn1(features)
            features = self.dropout(features)
            features = features.view(features.size(0), -1)
            features = self.fc1(features)
            features = self.bn2(features)
            features = F.normalize(features)
            if labels is not None:
                return self.margin(features, labels)
            return features


if __name__ == '__main__':
    model = EfficientNetArcFace(512, 1000, backbone='tf_efficientnet_b0_ns', pretrained=False, margin=0.5)
    model.cuda()
    # print(summary(model, (3, 128, 128)))
    model(torch.zeros((2, 3, 128, 128)).cuda(), torch.zeros((1, 1)).cuda())
