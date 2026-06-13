#!/usr/bin/env python3
"""

Usage:
  chmod +x pipeline.py
  ./pipeline.py \
    --image      ../../test/data/KITTI_data/new_data/training/image/000005.png \
    --pcd        ../../test/data/KITTI_data/new_data/training/pcd/000005.pcd \
    --mask       ../../test/data/KITTI_data/new_data/training/mask/000005.png \
    --refine-pcd ../../test/data/KITTI_data/new_data/training/folder_for_refinement/000005/poles_and_lane_markings.pcd \
    --calib-txt  ../../test/data/KITTI_data/new_data/training/calib/000005.txt \
    --lines2d    ../../test/results/results_Camera/kitti/training/000005.json \
    --lines3d    ../../test/results/results_cpp_pole/lane_pole/training/000005/3d_endpoints.json \
    --results-dir ../../test/results/PnL_projections_test_12052025 \
    --refine-bin ../cpp/build/refine
"""


import os
import sys
import json
import cv2
import numpy as np
import open3d as o3d
import itertools
import subprocess
import argparse
from scipy.optimize import least_squares
from cvxpnpl import pnl


CONFIG = {
    'CHECK_REFLECTION': True,
    'CHECK_BEHIND_FRAC': True,
    'MAX_BEHIND_FRAC': 0.65,
    'MIN_BEHIND_FRAC': 0.4,
    'CHECK_PARALLEL_LANES': False,
    'MAX_PARALLEL_ANGLE': 85.0,
    'USE_INFINITE_LINE_ERROR': True,
    'COMBINE_INFINITE_AND_SSE': False,
    'WEIGHT_INFINITE': 0.5,
    'WEIGHT_SSE': 0.5,
    'ENABLE_REFINEMENT': False,
    'REFINE_MAX_ITER': 100,
    'REFINE_METHOD': 'trf'
}


def read_2d_lines_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    clicked_lines = data.get("clicked_lines", [])
    lane_lines, pole_lines = [], []
    for L in clicked_lines:
        seg = [[L["x1"], L["y1"]], [L["x2"], L["y2"]]]
        (pole_lines if L["isPole"] else lane_lines).append(seg)
    return np.array(lane_lines, dtype=np.float32), np.array(pole_lines, dtype=np.float32)


def read_3d_lines_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    lane_lines_3d, pole_lines_3d = [], []
    for it in data:
        seg = [it["pmin"], it["pmax"]]
        if it.get("type","lane")=="pole":
            pole_lines_3d.append(seg)
        else:
            lane_lines_3d.append(seg)
    return np.array(lane_lines_3d, dtype=np.float32), np.array(pole_lines_3d, dtype=np.float32)


def read_kitti_intrinsics(calib_txt):
    P2 = None
    with open(calib_txt,'r') as f:
        for ln in f:
            if ln.startswith("P2:"):
                vals = list(map(float,ln.split()[1:]))
                P2 = np.array(vals).reshape(3,4)
                break
    if P2 is None:
        raise ValueError(f"Could not find 'P2:' in {calib_txt}")
    return P2[:3,:3]


def load_data(image_path, pcd_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not load image: {image_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    return img, np.asarray(pcd.points, dtype=np.float32)


# projection & cost fn
def transform_points(pts, R, t):
    return (R @ pts.T).T + t

def project_point_single(pt_3d, R, t, K):
    pc = R @ pt_3d + t
    if pc[2]<=0:
        return np.array([np.nan,np.nan],dtype=np.float32)
    uvw = K @ pc
    return uvw[:2]/uvw[2]


def line2d_normal_form(u1, u2):
    d = u2 - u1
    n = np.array([d[1], -d[0]],dtype=float)
    L = np.linalg.norm(n)
    if L<1e-12:
        return None, None
    n/=L
    c = n @ u1
    return n, c


def line_line_distance(n1,c1,n2,c2,angle_thresh=5.0):
    dotv = np.clip(n1@n2,-1,1)
    ang = min(np.degrees(np.arccos(dotv)), abs(180-np.degrees(np.arccos(dotv))))
    if ang < angle_thresh:
        return abs(c1-c2)
    return 999999.0


def infinite_line_distance_one_line(line_3d, line_2d, R, t, K):
    n_obs,c_obs = line2d_normal_form(line_2d[0], line_2d[1])
    if n_obs is None:
        return 999999.0
    p1_cam = R@line_3d[0] + t
    p2_cam = R@line_3d[1] + t
    if p1_cam[2]<=0 or p2_cam[2]<=0:
        return 999999.0
    uvw1 = K@p1_cam; uvw2 = K@p2_cam
    if uvw1[2]<=0 or uvw2[2]<=0:
        return 999999.0
    u1 = uvw1[:2]/uvw1[2]; u2 = uvw2[:2]/uvw2[2]
    n_proj,c_proj = line2d_normal_form(u1,u2)
    if n_proj is None:
        return 999999.0
    return line_line_distance(n_obs,c_obs,n_proj,c_proj)


def endpoint_sse_one_line(line_3d, line_2d, R, t, K):
    err=0.0
    for i in range(2):
        p3 = line_3d[i]; p2_obs = line_2d[i]
        p2_proj = project_point_single(p3,R,t,K)
        if np.any(np.isnan(p2_proj)):
            err+=1e6
        else:
            err += np.sum((p2_proj - p2_obs)**2)
    return err


def combined_cost_function(lanes_3d, lanes_2d, poles_3d, poles_2d, R, t, K):
    if CONFIG['COMBINE_INFINITE_AND_SSE']:
        total_inf = 0.0; total_sse = 0.0
        for i in range(len(lanes_3d)):
            total_inf += infinite_line_distance_one_line(lanes_3d[i], lanes_2d[i], R,t,K)
            total_sse += endpoint_sse_one_line(lanes_3d[i], lanes_2d[i], R,t,K)
        for i in range(len(poles_3d)):
            total_inf += infinite_line_distance_one_line(poles_3d[i], poles_2d[i], R,t,K)
            total_sse += endpoint_sse_one_line(poles_3d[i], poles_2d[i], R,t,K)
        return CONFIG['WEIGHT_INFINITE']*total_inf + CONFIG['WEIGHT_SSE']*total_sse
    else:
        if CONFIG['USE_INFINITE_LINE_ERROR']:
            total=0.0
            for i in range(len(lanes_3d)):
                total += infinite_line_distance_one_line(lanes_3d[i], lanes_2d[i], R,t,K)
            for i in range(len(poles_3d)):
                total += infinite_line_distance_one_line(poles_3d[i], poles_2d[i], R,t,K)
            return total
        else:
            total=0.0
            for i in range(len(lanes_3d)):
                total += endpoint_sse_one_line(lanes_3d[i], lanes_2d[i], R,t,K)
            for i in range(len(poles_3d)):
                total += endpoint_sse_one_line(poles_3d[i], poles_2d[i], R,t,K)
            return total


def pose_to_params(R, t):
    from scipy.spatial.transform import Rotation as Rsc
    e = Rsc.from_matrix(R).as_euler('xyz',degrees=False)
    return np.hstack((e,t))

def params_to_pose(p):
    from scipy.spatial.transform import Rotation as Rsc
    R = Rsc.from_euler('xyz', p[:3], degrees=False).as_matrix()
    return R, p[3:]

def residuals(x, lanes_3d, lanes_2d, poles_3d, poles_2d, K):
    R, t = params_to_pose(x)
    res = []
    for i in range(len(lanes_3d)):
        res.append(infinite_line_distance_one_line(lanes_3d[i], lanes_2d[i], R, t, K))
    for i in range(len(poles_3d)):
        res.append(infinite_line_distance_one_line(poles_3d[i], poles_2d[i], R, t, K))
    return np.array(res)

def refine_pose(R_init, t_init, lanes_3d, lanes_2d, poles_3d, poles_2d, K):
    x0 = pose_to_params(R_init, t_init)
    res = least_squares(lambda x: residuals(x, lanes_3d, lanes_2d, poles_3d, poles_2d, K),
                        x0, method=CONFIG['REFINE_METHOD'], max_nfev=CONFIG['REFINE_MAX_ITER'])
    return params_to_pose(res.x)



def main():
    p = argparse.ArgumentParser(description="Estimation + refinement pipeline")
    p.add_argument("--image",     required=True)
    p.add_argument("--pcd",       required=True)
    p.add_argument("--refine-pcd", required=True, help="Filtered PCD from lidar script output")
    p.add_argument("--mask",      required=True)
    p.add_argument("--calib-txt", required=True)
    p.add_argument("--lines2d",   required=True)
    p.add_argument("--lines3d",   required=True)
    p.add_argument("--results-dir",
                   default="../../test/results/PnL_projections",
                   help="where Pnl solver writes its output")
    p.add_argument("--refine-bin",
                   default="../cpp/build/refine",
                   help="path to refine executable")
    args = p.parse_args()

    # override constants
    IMAGE_PATH       = args.image
    PCD_PATH         = args.pcd
    REFINE_PCD      = args.refine_pcd
    MASK_PATH        = args.mask
    KITTI_CALIB_TXT  = args.calib_txt
    JSON_2D_PATH     = args.lines2d
    JSON_3D_PATH     = args.lines3d
    RESULTS_BASE_DIR = args.results_dir
    REFINE_BIN       = args.refine_bin

    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


    lanes_2d, poles_2d = read_2d_lines_from_json(JSON_2D_PATH)
    K_new             = read_kitti_intrinsics(KITTI_CALIB_TXT)
    lanes_3d, poles_3d = read_3d_lines_from_json(JSON_3D_PATH)
    img, pts_lidar    = load_data(IMAGE_PATH, PCD_PATH)

    
    H, W = img.shape[:2]
    log_data = {"solutions": [], "best_solution": {}}
    all_solutions = []
    sol_id = 0


    for p3d_l in itertools.permutations(lanes_3d,2):
      for p2d_l in itertools.permutations(lanes_2d,2):
        for p3d_p in itertools.permutations(poles_3d,1):
          for p2d_p in itertools.permutations(poles_2d,1):
            min3 = np.vstack([p3d_l, p3d_p])
            min2 = np.vstack([p2d_l, p2d_p])
            try:
                cands = pnl(line_2d=min2, line_3d=min3, K=K_new)
            except:
                continue
            for R_c, t_c in cands:
                sol_id += 1
                # reflection
                if CONFIG['CHECK_REFLECTION'] and np.linalg.det(R_c)<0:
                    discard="Reflect"; detR=np.linalg.det(R_c);
                else:
                    discard=""; detR=np.linalg.det(R_c)
                # behind frac
                pts_cam = transform_points(pts_lidar, R_c, t_c)
                bhFrac  = np.mean(pts_cam[:,2]<=0)
                if CONFIG['CHECK_BEHIND_FRAC'] and not(CONFIG['MIN_BEHIND_FRAC']<=bhFrac<=CONFIG['MAX_BEHIND_FRAC']):
                    discard += f"BehindFrac={bhFrac:.2f};"
                cost_val = combined_cost_function(lanes_3d, lanes_2d, poles_3d, poles_2d, R_c, t_c, K_new)
                # save reprojection image
                out_img = img.copy()
                uv = (K_new @ (R_c@pts_lidar.T + t_c.reshape(3,1))).T
                uv = uv[uv[:,2]>0]
                uv = uv[:,:2]/uv[:,2:3]
                for u,v in uv:
                    ui,vi = int(u),int(v)
                    if 0<=ui<W and 0<=vi<H:
                        cv2.circle(out_img,(ui,vi),1,(0,255,0),-1)
                sub = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
                fname = f"sol{sol_id}_cost{int(cost_val)}.png"
                subdir = os.path.join(RESULTS_BASE_DIR,sub)
                os.makedirs(subdir, exist_ok=True)
                cv2.imwrite(os.path.join(subdir,fname), out_img)
                # log
                entry = {"sol_id":sol_id,"detR":detR,"behindFrac":bhFrac,
                         "cost":cost_val,"discard":discard,
                         "R":R_c.tolist(),"t":t_c.tolist()}
                log_data["solutions"].append(entry)
                all_solutions.append((cost_val, discard, R_c, t_c))

    if not all_solutions:
        print("No solutions found. Exiting.")
        sys.exit(1)

    # pick best
    finals = [s for s in all_solutions if s[1]==""]
    if not finals:
        finals = all_solutions
    finals.sort(key=lambda x:x[0])
    best_cost, best_discard, best_R, best_t = finals[0]

    # save best 
    sub = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    subdir = os.path.join(RESULTS_BASE_DIR,sub)
    best_img = img.copy()
    uv = (K_new @ (best_R@pts_lidar.T + best_t.reshape(3,1))).T
    uv = uv[uv[:,2]>0]; uv=uv[:,:2]/uv[:,2:3]
    for u,v in uv:
        ui,vi = int(u),int(v)
        if 0<=ui<W and 0<=vi<H:
            cv2.circle(best_img,(ui,vi),3,(0,0,255),-1)
    cv2.imwrite(os.path.join(subdir,f"{sub}_best.png"), best_img)

    # write log 
    log_data["best_solution"] = {"cost":best_cost,"discard":best_discard,
                                  "R":best_R.tolist(),"t":best_t.tolist()}
    with open(os.path.join(RESULTS_BASE_DIR,f"{sub}.json"),'w') as f:
        json.dump(log_data,f,indent=2)
    print(f"Python PnL done. Results in {subdir}")


#-------------------------------------------------------------


    def write_intr_cpp(K, outp, cam_id="cam0"):
        payload={cam_id:{"param":{"cam_K":{"data":K.tolist()},
                                      "cam_dist":{"data":[[0,0,0,0,0]],"cols":5}}}}
        json.dump(payload, open(outp,'w'), indent=2)
    def write_ext_cpp(T4, outp, cam_id="cam0"):
        payload={cam_id:{"param":{"sensor_calib":{"data":T4.tolist()}}}}
        json.dump(payload, open(outp,'w'), indent=2)

    T4 = np.eye(4,dtype=float)
    T4[:3,:3] = best_R; T4[:3,3] = best_t
    intr_cpp = os.path.join(subdir,f"{sub}_intrinsic.json")
    ext_cpp  = os.path.join(subdir,f"{sub}_extrinsic.json")
    write_intr_cpp(K_new, intr_cpp)
    write_ext_cpp(T4,    ext_cpp)
    print(f"Wrote C++ JSONs: {intr_cpp}, {ext_cpp}")


    cmd = [REFINE_BIN, MASK_PATH, REFINE_PCD, intr_cpp, ext_cpp]
    print("Calling refine:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("C++ refinement complete.")

if __name__=="__main__":
    main()
