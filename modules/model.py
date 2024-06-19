import torch
import torch.nn as nn

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes, dropout_p=0.5, fc1_out=4096, fc2_out=4096, activation=nn.ReLU):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            activation(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            activation(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Classifier part with customizable parameters
        self.classifier = self._make_classifier(fc1_out, fc2_out, num_classes, dropout_p, activation)
    
    def _make_classifier(self, fc1_out, fc2_out, num_classes, dropout_p, activation):
        return nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 * 6 * 6, fc1_out),
            activation(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc1_out, fc2_out),
            activation(inplace=True),
            nn.Linear(fc2_out, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x