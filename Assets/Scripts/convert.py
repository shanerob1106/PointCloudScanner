import os
import torch
import torch.nn as nn
from PointNet import PointNetCustom


# --- Convert Model ---
def convert():

    # Config (Must match training)
    PYTORCH_MODEL_PATH = "./Assets/Scripts/best_segmentation_model.pth"
    ONNX_MODEL_PATH = "./Assets/Scripts/model.onnx"
    NUM_CLASSES = 2
    NUM_POINTS = 16384

    # Create the target directory if it doesn't exist
    os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)

    # Use CPU for conversion
    device = torch.device("cpu")

    # 1. Initialize the model and load the trained weights
    print(f"Loading model from {PYTORCH_MODEL_PATH}...")
    model = PointNetCustom(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 2. Create a dummy input tensor with the correct shape
    dummy_input = torch.randn(1, 3, NUM_POINTS, device=device)
    print(f"Created a dummy input tensor of shape {dummy_input.shape}.")

    # 3. Export the model to ONNX
    print(f"Exporting model to ONNX format at {ONNX_MODEL_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # 4. Verify the export
    if os.path.exists(ONNX_MODEL_PATH):
        print("\n-------------------------------------------")
        print("ONNX model conversion successful!")
        print(f"Model saved to: {ONNX_MODEL_PATH}")
        print("-------------------------------------------")
    else:
        print("Error: ONNX model conversion failed.")


if __name__ == "__main__":
    convert()
