import os
import glob
import random
import onnxruntime
import numpy as np
import open3d as o3d

# --- Run Inference Model ---
def run_inference():
    ONNX_MODEL_PATH = "./Assets/Scripts/model.onnx"

    # Randomly choose a file from the final dataset folder
    dataset_dir = "./Assets/Scripts/finalDataset"
    npz_files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    if not npz_files:
        print(f"No .npz files found in {dataset_dir}")
        return
    SAMPLE_DATA_PATH = random.choice(npz_files)
    print(f"Randomly selected sample: {SAMPLE_DATA_PATH}")

    # Make sure there is a valid model and sample file
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: ONNX model not found at {ONNX_MODEL_PATH}")
        print("Please run 'convert_to_onnx.py' first.")
        return

    if not os.path.exists(SAMPLE_DATA_PATH):
        print(f"Error: Sample data not found at {SAMPLE_DATA_PATH}")
        print("Please ensure your dataset is processed and available.")
        return

    # 1. Load the ONNX model
    print(f"Loading ONNX model from {ONNX_MODEL_PATH}...")
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model loaded. Input node: '{input_name}', Output node: '{output_name}'.")

    # 2. Load and prepare the sample data
    print(f"\nLoading sample data from {SAMPLE_DATA_PATH}...")
    data = np.load(SAMPLE_DATA_PATH)
    points = data["points"]

    # Prepare the input for the model
    input_tensor = np.expand_dims(points.T, axis=0).astype(np.float32)
    print(f"Prepared input tensor with shape: {input_tensor.shape}")

    # 3. Run Model
    print("\nRunning inference...")
    results = session.run([output_name], {input_name: input_tensor})

    output_logits = results[0]
    print("Inference complete.")

    # 4. Process the output
    predicted_labels = np.argmax(output_logits, axis=2).flatten()

    # 5. Visualize the result
    print("\nVisualizing segmentation result...")
    print("Green = Predicted Chair, Gray = Predicted Room")

    # Convert the points to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Label points the correct colour
    colors = np.zeros_like(points)
    colors[predicted_labels == 1] = [0.1, 0.8, 0.2]  # Green for chair
    colors[predicted_labels == 0] = [0.5, 0.5, 0.5]  # Gray for room
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Display the point cloud (Uncomment for raw labelled point cloud)
    # o3d.visualization.draw_geometries([pcd], window_name="ONNX Inference Result")

    # Display point cloud with mesh
    chair_points = pcd.select_by_index(np.where(predicted_labels == 1)[0])
    chair_points.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # Create a mesh using Alpha Shape (Best shape to visualize current model performance)
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(chair_points)
    for alpha in np.logspace(np.log10(0.027), np.log10(0.01), num=1):
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            chair_points, alpha, tetra_mesh, pt_map
        )
        mesh.compute_vertex_normals()

        convex_hull = mesh.compute_convex_hull()
        print(
            f"Convex hull has {len(convex_hull[0].triangles)} triangles. Volume: {convex_hull[0].get_volume()*10000} cm3 || Weight Estimate: {convex_hull[0].get_volume() * 1326.6}"
        )

        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # OTHER ATTEMPTS TO CREATE MESH (Various levels of success)
    # Uncomment bellow to see different meshing approaches

    # Create a mesh using Ball Pivoting
    # radii = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(chair_points, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([chair_points, mesh])

    # Create a mesh using Poisson Surface Reconstruction
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(chair_points)
    # o3d.visualization.draw_geometries([mesh], window_name="Mesh from Point Cloud")

# Mian loop
if __name__ == "__main__":
    run_inference()
