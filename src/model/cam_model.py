import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbone.googlenet_conv import GoogLeNetConv  
from src.layers.gap import GlobalAveragePooling
from src.head.classifier_head import ClassifierHead

class CAMModel(nn.Module):
    def __init__(self, backbone='googlenet', num_classes=1000):
        super().__init__()
        
        if backbone == 'alexnet':
            from src.backbone.alexnet_conv import AlexNetConv
            self.backbone = AlexNetConv()
            in_features = 256  
        elif backbone == 'vgg':
            from src.backbone.vgg_conv import VGGConv
            self.backbone = VGGConv()
            in_features = 512  
        else:
            self.backbone = GoogLeNetConv()
            in_features = 1024 

        self.gap = GlobalAveragePooling()
        self.classifier = ClassifierHead(in_features, num_classes)

    def forward(self, x):
        feature_maps = self.backbone(x) 
        
        gap_out = self.gap(feature_maps)  
        class_probs = self.classifier(gap_out)  

        return class_probs, feature_maps

    def get_cam(self, class_idx, feature_maps):
        weight_softmax = self.classifier.fc.weight 
        
        if isinstance(class_idx, int):
            w_c = weight_softmax[class_idx] 
            cam = torch.einsum('c,bchw->bhw', w_c, feature_maps)
        else:
            cam = []
            for i, c in enumerate(class_idx):
                w_c = weight_softmax[c]  
                cam_i = torch.einsum('c,chw->hw', w_c, feature_maps[i])
                cam.append(cam_i)
            cam = torch.stack(cam, dim=0)  
        
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1,1,1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam
