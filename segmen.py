import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class Segmentor(nn.Module):
    def __init__(self, num_classes=91):  
        super().__init__()
        self.model = deeplabv3_resnet50(pretrained=True)

        # Replacing the classifier head to match number of output classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.model(x)  # output is a dictionary
        return output['out']    # return only the segmentation map
