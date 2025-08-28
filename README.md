# Real-Time 3D Point Cloud Segmentation in Unity for Meta Quest

This repository contains a complete pipeline for performing real-time 3D semantic segmentation on point cloud data captured directly within a Meta Quest VR application. The system uses the Quest's scene understanding capabilities to scan the environment. It runs an AI model on-device using Unity Sentis to identify objects like a chair. The result is visualized as a segmented point cloud and a generated 3D mesh.

## Features

* **Real-time Environment Scanning**: Uses the Meta Quest's raycasting capabilities to capture point cloud data of the user's surroundings.
* **Intuitive VR Controls**: Scan, label, delete points, and trigger AI processing using the Oculus Touch controllers.
* **On-Device AI Inference**: A custom PointNet-like model, trained in PyTorch and converted to ONNX, runs directly on the headset via Unity Sentis for fast, local processing.
* **Semantic Segmentation**: Differentiates between two classes: **Chair** (Class 1) and **Room** (Class 0).
* **Dynamic Visualization**: Segmented points are color-coded in real-time within the VR environment.
* **3D Mesh Generation**: Creates a convex hull mesh from the segmented "chair" points using the `MIConvexHull` library.
* **Volume & Weight Estimation**: Calculates the volume of the generated mesh and provides a rough weight estimate.
* **Complete Python Pipeline**: Includes scripts for data preprocessing, augmentation, model training, and ONNX conversion.

## The Workflow

The project is divided into two main parts: a Python-based machine learning pipeline for model creation and a Unity application for real-world deployment.

1.  **Data Collection & Preparation (`preprocess.py`)**:
    * Initial point cloud data can be captured and saved as `.ply` files with labels directly from the Unity application.
    * `preprocess.py` reads these `.ply` files, applies augmentations like rotation, scaling, and jittering.
    * It then normalizes the coordinates, samples or pads the data to a fixed number of points (16,384), and saves the results as compressed `.npz` files.

2.  **Model Training (`train.py`)**:
    * The `PointNetCustom` model, defined in `PointNet.py`, is trained on the preprocessed `.npz` dataset.
    * The script uses a training/validation split, tracks accuracy and Mean Intersection over Union (mIoU), and saves the best-performing model weights as `best_segmentation_model.pth`.

3.  **Model Conversion (`convert.py`)**:
    * The trained PyTorch model (`.pth`) is converted into the ONNX format (`model.onnx`). This format is required for use with Unity Sentis.

4.  **Deployment & Inference (`PointCloudController.cs`)**:
    * The `model.onnx` asset is loaded into the Unity application.
    * The user scans their environment using the VR controllers.
    * The `PointCloudController.cs` script gathers the points, preprocesses them in a manner identical to the training pipeline, and feeds them into the Sentis inference engine.
    * The model's output, which contains the predicted labels for each point, is used to colorize the point cloud and generate the final mesh.

## Deep Dive: PointCloudController.cs

This is the core script that orchestrates the entire in-VR experience. It manages user input, data capture, visualization, and AI inference.

### State Management
The script operates on a simple state machine (`ScanState`) to guide the user through the process:
* `Idle`: The initial state where the user can begin scanning the chair.
* `ScanningChair`: The user holds the trigger to capture points belonging to the chair, which are visualized in green.
* `ScanningRoom`: After capturing the chair, the user scans the surrounding room geometry, visualized in white.
* `ReadyToProcess`: All scanning is complete. The user can now choose to save the raw data or run the AI segmentation.

### User Interaction & Controls
Input is handled via the `OVRInput` API:
* **Right Index Trigger**: Press and hold to scan points.
* **Left Index Trigger**: Press and hold to delete points within the indicator's radius.
* **'A' Button**: Advances the state (from chair scan to room scan, then to process). In the final state, it saves the point cloud.
* **'B' Button**: Runs the AI segmentation on the captured points.
* **Right Thumbstick Click**: Clears all scanned data and resets the application to the `Idle` state.

### Point Cloud Capture & Visualization
* A `scanIndicator` GameObject shows the user where they are pointing and the area that will be scanned.
* The `PerformScan` method uses a Fibonacci sphere sampling pattern to cast multiple rays, ensuring an even distribution of points.
* Captured points are stored in `_chairPoints` and `_roomPoints` lists.
* A `ParticleSystem` is used for efficient visualization of the thousands of captured points.

### Inference with Unity Sentis (`PointCloudInference` static class)
The `RunAISegmentation` method triggers the core AI logic, which is encapsulated in the static helper class `PointCloudInference`.
1.  **Preprocessing**: The captured `_combinedPoints` list is preprocessed. This function samples or duplicates points to match the model's required input size (`NUM_POINTS = 16384`). It also normalizes the points by centering them and scaling them to fit within a unit sphere.
2.  **Execution**: A `Tensor` is created and scheduled for execution on the GPU backend.
3.  **Output Processing**: The model's output tensor is read back, and the raw scores are converted into class labels (0 for room, 1 for chair).

### Mesh Generation and Analysis
* After inference, `VisualizeInferenceResult` is called.
* It filters the points predicted as "chair" and passes them to `CreateMeshFromPoints`.
* This method uses the **MIConvexHull** library to generate a 3D convex hull mesh from the chair points.
* The generated mesh is added to the scene with a semi-transparent material.
* Finally, `CalculateMeshVolume` computes the volume of the mesh, which is then used to estimate its weight.

## Python ML Pipeline

The Python scripts provide the necessary tools to train the model deployed in the Unity app.

* `PointNet.py`: Defines the neural network architecture, featuring sequential layers for feature extraction (`sa`) and feature propagation (`fp`).
* `PointCloudDataset.py`: A standard PyTorch `Dataset` class that loads the `.npz` files created by the preprocessing script.
* `preprocess.py`: This script reads raw `.ply` data, applies augmentations (rotation, scaling, jitter), and saves it in a format ready for training.
* `train.py`: The main training script. It handles data loading, model training, validation, and saving the best model based on mIoU.
* `run.py`: A utility script to test the exported `model.onnx` using `onnxruntime` and visualize the output with Open3D.

## Setup and Usage

### Prerequisites
* **Unity**: Version 2022.3.x or newer.
* **Unity Packages**:
    * Meta XR SDK (Oculus Integration)
    * Unity Sentis
    * `MIConvexHull` library
* **Python**: Version 3.8+.
* **Python Libraries**: `torch`, `numpy`, `onnx`, `onnxruntime`, `open3d`, `tqdm`, `scikit-learn`, `plyfile`.

### How to Run
1.  **Train the Model**:
    * Place your raw `.ply` data in the `Assets/Scripts/dataset` folder.
    * Run `python preprocess.py` to generate the training data in `finalDataset`.
    * Run `python train.py` to train the model. This will create `best_segmentation_model.pth`.
    * Run `python convert.py` to create `model.onnx`.
2.  **Unity Project**:
    * Move the generated `model.onnx` file into your Unity project's `Assets` folder.
    * Assign the `model.onnx` file to the `modelAsset` field in the `PointCloudController` component in the Inspector.
    * Set up your scene with the necessary Meta Quest prefabs (e.g., OVRCameraRig).
    * Attach `PointCloudController.cs` to a GameObject and link the required components (e.g., `rayOriginAnchor`, `scanParticles`).
3.  **Build and Deploy**:
    * Build the project for the Android platform and deploy it to your Meta Quest headset.
    * Follow the in-app controls listed above to scan and segment objects.
