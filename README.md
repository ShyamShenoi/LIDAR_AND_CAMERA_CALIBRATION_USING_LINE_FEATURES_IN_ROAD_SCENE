# Automatic LiDAR Camera Calibration using Line Features in Road Scene

A targetless automatic extrinsic calibration pipeline for LiDAR and camera sensors. Instead of a checkerboard or dedicated calibration target, the method extracts naturally occurring road scene features — **lane markings** and **vertical poles** — and uses them as shared line correspondences to estimate the sensor-to-sensor transformation.

---

## How It Works

The pipeline runs in five stages:

```
Camera Image + LiDAR Point Cloud
         |
         |-- [1] Lane Mask Generation  (DeepLab V3 + DenseCRF)
         |         src/my_thesis_lane_mask/
         |
         |-- [2] 2D Line Detection     (PCA + vanishing point filter)
         |         src/python/camera1.py
         |
         |-- [3] 3D Line Detection     (RANSAC plane + Otsu/percentile intensity + cylinder segmentation)
         |         src/cpp/lidar  (binary)
         |
         |-- [4] PnL Estimation        (Perspective-n-Line via cvxPnPl, over permuted line pairs)
         |         src/python/pipeline.py
         |
         `-- [5] Refinement            (random search maximising LiDAR overlap on lane/pole mask)
                   src/cpp/refine  (binary)
```

**Stage 1 — Lane Mask:** DeepLab V3 produces a coarse road segmentation; a Conditional Random Field (CRF) sharpens boundaries to pixel-level accuracy.

**Stage 2 — 2D Lines:** Connected components are extracted from the mask, PCA fits a line to each component, and a vanishing-point filter keeps only geometrically consistent lane and pole candidates.

**Stage 3 — 3D Lines:** The ground plane is segmented via RANSAC. Lane markings are isolated using an adaptive intensity threshold (max of Otsu's method and the 97.5th percentile of the ground plane point cloud). Poles are detected above the ground plane using Euclidean clustering followed by cylinder model fitting.

**Stage 4 — PnL Estimation:** Matched 2D-3D line pairs are fed to the [cvxPnPl](https://github.com/cvlab-epfl/cvxpnpl) convex solver. Candidate solutions are scored using an infinite-line reprojection error and the best valid pose is selected.

**Stage 5 — Refinement:** Starting from the PnL estimate, a multi-scale random search perturbs the 6-DOF pose and keeps updates that increase the fraction of LiDAR feature points landing on the lane/pole mask.

---

## Project Structure

```
.
├── src/
│   ├── cpp/
│   │   ├── lidar.cpp          # 3D line detection (lane markings + poles)
│   │   ├── refinement.cpp     # Mask-overlap refinement via random search
│   │   └── CMakeLists.txt
│   ├── python/
│   │   ├── camera1.py         # 2D line detection from lane mask
│   │   └── pipeline.py        # PnL estimation + orchestration
│   ├── config/
│   │   ├── lidar_kitti_params.yaml      # Parameters for KITTI dataset
│   │   └── lidar_sensorcar_params.yaml  # Parameters for custom sensor rig
│   └── my_thesis_lane_mask/   # Lane mask generation tool (see its own README)
└── requirements.txt
```

---

## Requirements

### Python

```bash
pip install -r requirements.txt
```

Key dependencies: `opencv-python`, `open3d`, `numpy`, `scipy`, `cvxpnpl`, `scikit-image`.

### C++

- CMake (3.10+)
- C++14 compatible compiler
- [PCL](https://pointclouds.org/) (Point Cloud Library)
- OpenCV
- yaml-cpp
- jsoncpp

---

## Build

```bash
cd src/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This produces two binaries inside `build/`:
- `lidar` — runs 3D line detection on a point cloud
- `refine` — runs mask-overlap refinement given an initial extrinsic estimate

---

## Running the Pipeline

### Step 1 — Generate the lane mask

See [`src/my_thesis_lane_mask/README.md`](src/my_thesis_lane_mask/README.md) for full instructions. The output is a binary PNG mask of lane markings and road boundaries.

### Step 2 — Detect 2D lines from the mask

```bash
python3 src/python/camera1.py \
    --image /path/to/mask.png \
    --output /path/to/2d_endpoints.json
```

An interactive window opens showing detected lines. Left-click to select the lines you want to use as calibration features, then press `q` to save and exit.

Add `--debug` to show intermediate processing windows.

### Step 3 — Detect 3D lines from the point cloud

```bash
./src/cpp/build/lidar \
    /path/to/scan.pcd \
    src/config/lidar_kitti_params.yaml \
    /path/to/output_dir
```

Outputs `lane_markings.pcd`, `poles_detected.pcd`, `poles_and_lane_markings.pcd`, and `3d_endpoints.json` into the specified output directory. An interactive 3D viewer opens — click lines to record their 3D endpoints.

### Step 4 & 5 — PnL estimation + refinement

```bash
python3 src/python/pipeline.py \
    --image    /path/to/image.png \
    --pcd      /path/to/scan.pcd \
    --mask     /path/to/mask.png \
    --refine-pcd /path/to/output_dir/poles_and_lane_markings.pcd \
    --calib-txt  /path/to/calib.txt \
    --lines2d    /path/to/2d_endpoints.json \
    --lines3d    /path/to/output_dir/3d_endpoints.json \
    --results-dir /path/to/results \
    --refine-bin  src/cpp/build/refine
```

The script writes per-solution reprojection images and a JSON log to `--results-dir`, then calls the C++ refinement binary on the best solution. The final extrinsic matrix is saved as `<image_name>_extrinsic.txt`.

---

## Configuration

All LiDAR detection parameters are controlled via the YAML config files in `src/config/`. Key parameters:

| Parameter | Description |
|---|---|
| `auto_percentile` | Percentile used alongside Otsu's method for adaptive intensity thresholding |
| `lane_plane_distance_threshold` | RANSAC inlier distance for ground plane segmentation |
| `lane_max_lines` | Maximum number of lane lines to extract |
| `cylinder_min/max_radius` | Radius bounds for pole cylinder fitting |
| `pole_min_height` | Minimum vertical extent for a valid pole |

---

## Lane Mask Tool

The `src/my_thesis_lane_mask/` subfolder is a standalone tool for generating lane masks using DeepLab V3 and DenseCRF. It is designed to run on Google Colab (T4 GPU). See its [README](src/my_thesis_lane_mask/README.md) for setup and execution instructions.
