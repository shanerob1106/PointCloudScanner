# Voxel Grid Averaging Implementation - Automatic Pipeline

## Overview
Added voxel grid downsampling functionality with 2cm (0.02m) voxels to reduce point cloud density while preserving spatial structure. The system now features an **automatic preprocessing pipeline** that runs when the user presses 'B' for AI segmentation.

## New Automatic Workflow

### User Experience:
1. **Scan Chair** (hold Right Trigger, press 'A' when done)
2. **Scan Room** (hold Right Trigger, press 'A' when done)  
3. **Press 'B'** → **Automatic Pipeline Executes:**
   - Step 1: Statistical Outlier Removal (SOR)
   - Step 2: Voxel Grid Averaging (2cm)
   - Step 3: AI Model Inference
   - Real-time visualization updates after each step

### Controls in ReadyToProcess State:
- **'A'** = Save point cloud
- **'B'** = AI Segmentation (auto SOR + voxel grid + inference) *(UPDATED)*
- **'X'** = Manual Statistical Outlier Removal *(for testing/debugging)*

## Technical Implementation

### Automatic Pipeline (`RunAISegmentation` method):

#### Step 1: Statistical Outlier Removal
- Removes statistical outliers from the combined point cloud
- Updates `_combinedPoints` and `_labels` with filtered data
- Updates particle visualization to show cleaned points
- Logs reduction statistics

#### Step 2: Voxel Grid Averaging (2cm)
- Applies 2cm voxel grid downsampling to the filtered points
- Uses spatial heuristics to preserve chair/room labels
- Updates `_combinedPoints` and `_labels` with downsampled data  
- Updates particle visualization to show voxel-averaged points
- Logs reduction statistics

#### Step 3: AI Model Inference
- Runs the neural network on the preprocessed point cloud
- `PreprocessDataWithMapping` now skips automatic SOR/voxel grid (done manually)
- Performs only coordinate conversion, centering, normalization, and sampling
- Visualizes final AI predictions with color coding

### Modified Methods:

#### `RunAISegmentation()`:
- Now implements 3-step automatic pipeline
- Provides detailed logging for each step
- Updates visualization after each preprocessing step
- Shows comprehensive statistics at completion

#### `PreprocessDataWithMapping()`:
- **Simplified**: No longer performs automatic SOR or voxel grid
- Works directly with input points (assumes preprocessing already done)
- Creates 1:1 point mapping for cleaner data flow
- Focuses only on model-specific transformations

### Performance Benefits:

1. **Streamlined UX**: Single button press for complete pipeline
2. **Visual Feedback**: Users see each preprocessing step in real-time
3. **Optimal Processing**: SOR → Voxel Grid → Model (logical order)
4. **Clear Logging**: Detailed statistics for each step
5. **Memory Efficiency**: Intermediate results properly managed

## GPU vs CPU Implementation

Both SOR and Voxel Grid support GPU acceleration:
- **GPU Mode**: Uses compute shaders for parallel processing
- **CPU Fallback**: Dictionary/list-based implementations
- **Automatic Selection**: Based on compute shader availability

## Backward Compatibility

- Manual outlier removal ('X' button) still available for debugging
- All existing save functionality unchanged
- Clear scan functionality works as before
- API methods preserved for programmatic access

## Example Log Output:
```
Running AI segmentation with automatic preprocessing pipeline...
Step 1/3: Statistical Outlier Removal
GPU Statistical Outlier Removal completed in 45ms
SOR: 8247 -> 7823 points (removed 424 outliers)

Step 2/3: Voxel Grid Averaging (2cm)  
GPU Voxel Grid Downsampling completed in 23ms
Voxel Grid: 7823 -> 3156 points (reduced by 4667 points)

Step 3/3: AI Model Inference
Inference complete. Original points: 8247, Kept after preprocessing: 3156

=== AI Segmentation Pipeline Complete ===
Original points: 8247
After SOR + Voxel Grid: 3156  
Final inference points: 3156
Total reduction: 5091 points
```

This implementation provides a smooth, automated workflow that maximizes both performance and user experience in VR environments.
