# Automatic Calibration Thesis Framework

This repository provides a self-contained pipeline for calibrating camera and LiDAR using detected pole and lane markers. It features a complete workflow starting from image and point cloud processing down to Perspective-n-Line (PnL) matrix estimation and refinement.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Building the C++ Lidar Tools](#building-the-c-lidar-tools)
- [Running the Pipeline](#running-the-pipeline)

## Project Structure
- `src/cpp`: Contains downsampling, segmentation, and clustering point-cloud utilities (e.g., `lidar.cpp` and `refinement.cpp`).
- `src/python`: Contains scripts like `camera1.py` for image line extraction, and `pipeline.py` which interfaces OpenCV, PCL tools, and cvxPnpl algorithms.
- `src/config`: Contains YAML configurations for `lidar.cpp` execution.
- `src/my_thesis_lane_mask`: Tool for automated lane mask extraction from camera images using DeepLab V3 and DenseCRF.

## Requirements
To run the python pipeline, you will need:
- Standard tools: Python 3.x, CMake, a C++14 capable compiler.
- Install the python requirements via:
  ```bash
  pip install -r requirements.txt
  ```
For the C++ portion, you must have:
- `PCL` (Point Cloud Library) 
- `OpenCV` 
- `yaml-cpp`

## Building the C++ Lidar Tools
```bash
cd src/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```
This produces the `lidar` and `refine` binaries required by the python pipeline.

## Running the Pipeline
You can run the full tool using the pipeline script (adjust paths as necessary):
```bash
python3 src/python/pipeline.py \
    --image /path/to/image.png \
    --pcd /path/to/sync.pcd \
    --mask /path/to/mask.png \
    --refine-pcd /path/to/poles_and_lane_markings.pcd \
    --calib-txt /path/to/calib.txt \
    --lines2d /path/to/2d_endpoints.json \
    --lines3d /path/to/3d_endpoints.json
```
