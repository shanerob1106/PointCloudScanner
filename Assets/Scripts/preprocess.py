import os
import numpy as np
import open3d as o3d
from plyfile import PlyData
import multiprocessing
import copy

# --- Configuration ---
SOURCE_DIR = "./Assets/Scripts/dataset"
TARGET_DIR = "./Assets/Scripts/finalDataset"
AUGMENTATIONS_PER_FILE = 29
TOTAL_POINTS = 16384


# Apply a random rotation to the point cloud
def rotate_point_cloud(points):
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    psi = np.random.uniform(0, 2 * np.pi)

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(psi), -np.sin(psi)], [0, np.sin(psi), np.cos(psi)]]
    )

    Ry = np.array(
        [[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]
    )

    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    rotation_matrix = Rz @ Ry @ Rx
    return points @ rotation_matrix.T


# Randomly scale the point cloud
def scale_point_cloud(points):
    scale = np.random.uniform(0.85, 1.15)
    return points * scale


# Add some jitter to the points
def jitter_point_cloud(points, sigma=0.005, clip=0.02):
    N, C = points.shape
    assert C == 3
    jitter = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return points + jitter


# Apply the transforms and save the new point cloud
def process_and_save(points, labels, target_path):

    # 1. Flip the Z-axis
    points[:, 2] = -points[:, 2]

    # 2. Convert points to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 3. Convert the PointCloud points to Numpy Array
    processed_points = np.asarray(pcd.points)

    if len(processed_points) == 0:
        return

    # 4. Random Sampling / Padding
    num_available = len(processed_points)
    replace = num_available < TOTAL_POINTS
    chosen_indices = np.random.choice(num_available, TOTAL_POINTS, replace=replace)
    final_points = processed_points[chosen_indices]
    final_labels = labels[chosen_indices]

    # 5. Coordinate Transformations (Normalization)
    final_points -= np.mean(final_points, axis=0)
    max_distance = np.max(np.linalg.norm(final_points, axis=1))
    if max_distance > 1e-6:
        final_points /= max_distance

    # 6. Save compressed output
    np.savez_compressed(
        target_path,
        points=final_points.astype(np.float32),
        labels=final_labels.astype(np.int64),
    )


# Process a file at a time to include a new augmentation
def process_file_and_generate_augmentations(source_path, base_target_path):
    try:
        # 1. Load the raw data and labels
        plydata = PlyData.read(source_path)
        vertex = plydata["vertex"]
        points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
        labels = np.array(vertex["label"], dtype=np.int64)

        # 2. Process the original point cloud
        process_and_save(copy.deepcopy(points), copy.deepcopy(labels), base_target_path)

        # 3. Generate and save augmented versions
        for i in range(AUGMENTATIONS_PER_FILE):
            augmented_points = copy.deepcopy(points)

            # Apply a random sequence of augmentations
            augmented_points = rotate_point_cloud(augmented_points)
            augmented_points = scale_point_cloud(augmented_points)
            augmented_points = jitter_point_cloud(augmented_points)

            # Create a new file path for the augmented data
            dir_name = os.path.dirname(base_target_path)
            base_name = os.path.splitext(os.path.basename(base_target_path))[0]
            augmented_target_path = os.path.join(dir_name, f"{base_name}_aug_{i+1}.npz")

            # Process and save the augmented point cloud
            process_and_save(
                augmented_points, copy.deepcopy(labels), augmented_target_path
            )

    except Exception as e:
        print(f"Error processing {os.path.basename(source_path)}: {e}")


# Main loop
if __name__ == "__main__":
    os.makedirs(TARGET_DIR, exist_ok=True)

    # List of files to process
    files_to_process = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".ply")]

    if not files_to_process:
        print(f"Error: No .ply files found in '{SOURCE_DIR}'.")
    else:
        # Get number of cpus cores available
        num_workers = multiprocessing.cpu_count()
        print(
            f"Found {len(files_to_process)} files. Starting processing with {num_workers} worker processes..."
        )
        print(
            f"Each file will be converted to 1 original + {AUGMENTATIONS_PER_FILE} augmented samples."
        )

        # Create tasks for each file, name, and save in target location
        tasks = []
        for filename in files_to_process:
            source_path = os.path.join(SOURCE_DIR, filename)
            target_filename = os.path.splitext(filename)[0] + ".npz"
            target_path = os.path.join(TARGET_DIR, target_filename)
            tasks.append((source_path, target_path))

        # Assign each of the tasks to the workers/cpu cores
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(process_file_and_generate_augmentations, tasks)

        # Final count output to ensure correct amount has been made
        final_file_count = len(os.listdir(TARGET_DIR))
        print(
            f"\nProcessing complete! Final dataset contains {final_file_count} files in '{TARGET_DIR}'."
        )
