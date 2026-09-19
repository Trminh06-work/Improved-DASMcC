import torch
import torch.nn as nn



class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels = 12, out_channels = 24, kernel_size = 7, padding = 3, stride = 2, bias = False),
            nn.BatchNorm1d(24),
            nn.ReLU(),

            nn.Conv1d(in_channels = 24, out_channels = 48, kernel_size = 7, padding = 3, stride = 2, bias = False),
            nn.BatchNorm1d(48),
            nn.ReLU(),

            nn.Conv1d(in_channels = 48, out_channels = 96, kernel_size = 5, padding = 2, stride = 2, bias = False),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )

        self.fusion = nn.Conv1d(in_channels = 96, out_channels = 1, kernel_size = 1, stride = 1)


    def forward(self, x):
        x = self.features(x)       # B x 96 x 125
        x = self.fusion(x)         # B x 1 x 125
        return x.squeeze(1)        # B x 125



class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.feature_extractor = CNNFeatureExtractor()  # B x 125

        self.classifier = nn.Sequential(
            nn.Linear(125, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x