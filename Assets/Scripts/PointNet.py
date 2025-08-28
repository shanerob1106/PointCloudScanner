import os
import torch
import torch.nn as nn

# --- Neural Network Model --- 
class PointNetCustom(nn.Module):
    def __init__(self, num_classes):
        super(PointNetCustom, self).__init__()

        # Feature Extraction (Set Abstraction) Layers
        self.sa1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), 
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.sa2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.sa3 = nn.Sequential(
            nn.Conv1d(128, 256, 1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.sa4 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Feature Propagation Layers
        self.fp1 = nn.Sequential(
            nn.Conv1d(512 + 256, 256, 1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fp2 = nn.Sequential(
            nn.Conv1d(256 + 128, 256, 1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fp3 = nn.Sequential(
            nn.Conv1d(128 + 64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Final Prediction Head
        self.final_conv = nn.Sequential(
            nn.Conv1d(128, 128, 1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, x):
        # Feature Extraction layers
        l1_features = self.sa1(x)
        l2_features = self.sa2(l1_features)
        l3_features = self.sa3(l2_features)
        l4_features = self.sa4(l3_features)
        
        # Global feature extraction
        global_feature = torch.max(l4_features, 2, keepdim=True)[0]
        global_feature_expanded = global_feature.repeat(1, 1, x.size(2))
        
        # Feature Propogation input layer using global features
        fp1_input = torch.cat((global_feature_expanded, l3_features), dim=1) 
        d1_features = self.fp1(fp1_input)
        
        fp2_input = torch.cat((d1_features, l2_features), dim=1)
        d2_features = self.fp2(fp2_input)
        
        fp3_input = torch.cat((d2_features, l1_features), dim=1)
        d3_features = self.fp3(fp3_input)

        # Final output layer
        logits = self.final_conv(d3_features)
        return logits.transpose(1, 2)