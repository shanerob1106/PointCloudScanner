import os
import glob
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from PointNet import PointNetCustom
from torch.utils.data import DataLoader
from PointCloudDataset import PointCloudDataset
from sklearn.model_selection import train_test_split


# --- IoU Calculation ---
def calculate_iou(preds, labels, num_classes):
    iou_list = []

    # Get the predict label indexs for each class
    for cls in range(num_classes):
        pred_inds = preds == cls
        label_inds = labels == cls

        # Calculate IoU
        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union
        iou_list.append(iou)

    # Find the mean IoU
    valid_ious = [iou for iou in iou_list if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    return iou_list, mean_iou


# --- Training loop ---
def train():

    # Config
    DATASET_DIR = "./Assets/Scripts/finalDataset/"
    NUM_CLASSES = 2
    BATCH_SIZE = 12
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    MODEL_SAVE_PATH = "./Assets/Scripts/best_segmentation_model.pth"

    # Check to see if Cuda (nVidia GPU) devices is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train/Validation Split
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files:
        raise ValueError(f"No .npz files found in directory: {DATASET_DIR}")

    # 80/20 split train/test data
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    print(
        f"Found {len(all_files)} total files: {len(train_files)} training, {len(val_files)} validation."
    )

    train_dataset = PointCloudDataset(file_paths=train_files)
    val_dataset = PointCloudDataset(file_paths=val_files)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Loss, Optimizer, best mIoU
    model = PointNetCustom(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_miou = 0.0

    # Epoch loop
    for epoch in range(EPOCHS):

        # Set model to train
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_points = 0

        # TQDM progress bar tracker to visualize training progress
        train_progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"
        )
        for points, labels in train_progress_bar:
            points, labels = points.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(points)

            outputs = outputs.reshape(-1, NUM_CLASSES)
            labels = labels.reshape(-1)

            # Loss calculation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Total loss based on correct points labelled
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train_points += labels.size(0)
            total_train_correct += (predicted == labels).sum().item()

            train_progress_bar.set_postfix(
                loss=loss.item(), acc=f"{(total_train_correct/total_train_points):.4f}"
            )

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_correct / total_train_points

        # Set model to evaluate
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_points = 0
        all_val_preds = []
        all_val_labels = []

        # Disable gradient calculation
        with torch.no_grad():

            # TQDM progress bar tracker to visualize validation progress
            val_progress_bar = tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"
            )
            for points, labels in val_progress_bar:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)

                # Get the predicted label indices for each class
                outputs = outputs.reshape(-1, NUM_CLASSES)
                labels = labels.reshape(-1)

                # Loss calculation
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val_points += labels.size(0)
                total_val_correct += (predicted == labels).sum().item()

                all_val_preds.append(predicted.cpu())
                all_val_labels.append(labels.cpu())

                val_progress_bar.set_postfix(
                    loss=loss.item(), acc=f"{(total_val_correct/total_val_points):.4f}"
                )

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_acc = total_val_correct / total_val_points

        # Find IoU
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        iou_list, mean_iou = calculate_iou(all_val_preds, all_val_labels, NUM_CLASSES)

        # Epoch output
        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f} | "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}, Val mIoU={mean_iou:.4f}"
        )

        # Save Best Model by mIoU
        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved (mIoU={best_miou:.4f})")

    print(
        f"Training complete. Best mIoU={best_miou:.4f}. Model saved to {MODEL_SAVE_PATH}"
    )


# Main loop
if __name__ == "__main__":
    train()
