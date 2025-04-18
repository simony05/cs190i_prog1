import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv1_model(nn.Module):
    def __init__(self, split_size = 9, num_boxes = 4, num_classes = 20):
        super().__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # LAYER 1
        layer_1 = nn.Sequential(
            # (kernel size, output filters, stride, padding)
            # Conv2d(in_channels, out_channels, kernels, stride, padding)
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            # BatchNorm2d normalizes data to stabilize training and help prevent overfitting
            nn.BatchNorm2d(64),
            # LeakyReLU introduces nonlinearity to update neurons even when gradient is small or negative
            nn.LeakyReLU(0.1),
            # MaxPool2d takes maximum value in each window
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        # LAYER 2
        layer_2 = nn.Sequential(
            # (3, 192, 1, 1)
            nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        # LAYER 3
        layer_3 = nn.Sequential(
            # (1, 128, 1, 0)
            nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # (3, 256, 1, 1)
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            # (1, 256, 1, 0)
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            # (3, 512, 1, 1)
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        # LAYER 4
        # blocks (1, 256, 1, 0) and (3, 512, 1, 1) repeat 4 times
        layer_4 = []
        for _ in range(4):
            layer_4 += [
                # (1, 256, 1, 0)
                nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                # (3, 512, 1, 1)
                nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
            ]

        layer_4 += [
            # (1, 512, 1, 0)
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            # (3, 1024, 1, 1)
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        ]

        # LAYER 5
        # blocks (1, 512, 1, 0) and (3, 1024, 1, 1) repeat 2 times
        layer_5 = []
        for _ in range(2):
            layer_5 += [
                # (1, 512, 1, 0)
                nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                # (3, 1024, 1, 1)
                nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
            ]

        layer_5 += [
            # (3, 1024, 1, 1)
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            # (3, 1024, 2, 1)
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        ]

        # LAYER 6
        layer_6 = nn.Sequential(
            # (3, 1024, 1, 1)
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            # (3, 1024, 1, 1)
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        self.features = nn.Sequential(
            *layer_1,
            *layer_2,
            *layer_3,
            *layer_4,
            *layer_5,
            *layer_6,
        )

        # fully connected layers
        self.fcl = nn.Sequential(
            # flattens CNN feature map output into 1-D vector
            nn.Flatten(),
            # maps flattened features to hidden layer of 496 units
            nn.Linear(1024 * split_size * split_size, 496),  # YOLOv1 paper uses 4096
            # no neurons dropped in training
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            # final prediction layer
            nn.Linear(496, split_size * split_size * (num_classes + num_boxes * 5)), # num_classes + boxes * 5 (p, x, y, h, w)
        )

    def forward(self, x):
        x = self.features(x)
        # upsample from 7x7 → 9x9
        x = F.interpolate(x, size = (self.split_size, self.split_size), mode = "bilinear", align_corners = False)
        return self.fcl(torch.flatten(x, start_dim=1))