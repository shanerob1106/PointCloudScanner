import os
import glob
import torch
import numpy as np
import torch.utils.data as Dataset

# --- Dataset Class ---
class PointCloudDataset(Dataset):
    # Load and process the datasets from npz format
    def __init__(self, data_dir=None, file_paths=None):
        if file_paths is None:
            if data_dir is None:
                raise ValueError("Either data_dir or file_paths must be provided.")
            self.file_paths = glob.glob(os.path.join(data_dir, '*.npz'))
        else:
            self.file_paths = file_paths
        
        if not self.file_paths:
            raise ValueError(f"No .npz files found.")

    def __len__(self):
        return len(self.file_paths)

    # Get each of the point clouds and load the data into correct tensors
    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        points = data['points']
        labels = data['labels']

        # Simplified PointNet++ expects (3, N)
        points_tensor = torch.from_numpy(points.T).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        return points_tensor, labels_tensor