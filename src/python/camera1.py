#!/usr/bin/env python3

import cv2
import numpy as np
import json
import time
import math
from skimage.morphology import skeletonize
import itertools   
import argparse




# GLOBAL CONFIG

MIN_REGION_AREA          = 100      
MAX_GAP_BETWEEN_SEGMENTS = 100      
SLOPE_DIFF_THRESHOLD     = 0.5      

POLE_SLOPE_THRESHOLD     = 3.0      
PCA_RATIO_THRESHOLD      = 3.0      
MIN_LANE_SLOPE_THRESHOLD = 0.7

VERTICAL_BAND_RATIO      = 0.3


VP_ANG_THRESHOLD         = 40.0     


# MORPHOLOGI

def morphological_cleanup(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    cleaned = cv2.dilate(cleaned, kernel2, iterations=1)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel3, iterations=1)
    return cleaned


def skeletonize_mask(mask):
    mask_bin = (mask>127).astype(np.uint8)
    skel = skeletonize(mask_bin).astype(np.uint8)*255
    return skel


# VANISHINGPOINT

def intersect(l1, l2):
    x1,y1,x2,y2,_,_ = l1
    x3,y3,x4,y4,_,_ = l2
    D = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(D) < 1e-6:
        return None
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / D
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / D
    return (Px, Py)

def estimate_vanishing_point(lines):
    pts = []
    for a, b in itertools.combinations(lines, 2):
        p = intersect(a, b)
        if p:
            pts.append(p)
    if not pts:
        return None
    xs, ys = zip(*pts)
    return (np.median(xs), np.median(ys))

def filter_by_vp(lines, vp, ang_th=VP_ANG_THRESHOLD):
    kept = []
    ang_th = math.radians(ang_th)
    for ln in lines:
        x1,y1,x2,y2,slope,isPole = ln
        if not isPole and abs(slope) < MIN_LANE_SLOPE_THRESHOLD:
            if DEBUG:
                print((f"[SLOPE_DROP] dropping lane slope={slope:.2f}"))
            continue


        mid = np.array([ (x1+x2)/2, (y1+y2)/2 ])
        dir_vec = np.array([ x2-x1, y2-y1 ], float)
        to_vp   = np.array([ vp[0]-mid[0], vp[1]-mid[1] ], float)
        cosang  = np.dot(dir_vec, to_vp) / (
            np.linalg.norm(dir_vec)*np.linalg.norm(to_vp) + 1e-9
        )
        angle = abs(math.acos(np.clip(cosang, -1, 1)))
        if angle < ang_th:
            kept.append(ln)
        elif DEBUG:
            print(f"[VP] dropping line at mid={mid} angle={math.degrees(angle):.1f}°")
    return kept


def should_merge(lineA, lineB):
    # if type differ => no
    if lineA[-1] != lineB[-1]:
        return False
    slopeA = lineA[4]
    slopeB = lineB[4]
    if abs(slopeA) > 1e6:
        slopeA = math.inf
    if abs(slopeB) > 1e6:
        slopeB = math.inf


    if slopeA == math.inf and slopeB == math.inf:
        slope_ok = True
    elif slopeA == math.inf or slopeB == math.inf:
        slope_ok = False
    else:
        slope_ok = abs(slopeA - slopeB) < SLOPE_DIFF_THRESHOLD
    if not slope_ok:
        return False

    
    if not lineA[-1] and not lineB[-1]:
        if abs(slopeA) < 0.5 and abs(slopeB) < 0.5:
            midA = ((lineA[0]+lineA[2])/2, (lineA[1]+lineA[3])/2)
            midB = ((lineB[0]+lineB[2])/2, (lineB[1]+lineB[3])/2)
            if abs(midA[0]-midB[0]) > (MAX_GAP_BETWEEN_SEGMENTS*0.5):
                return False


    (xA1,yA1,xA2,yA2,_,_) = lineA
    (xB1,yB1,xB2,yB2,_,_) = lineB
    def dist(a,b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    for ea in [(xA1,yA1),(xA2,yA2)]:
        for eb in [(xB1,yB1),(xB2,yB2)]:
            if dist(ea, eb) < MAX_GAP_BETWEEN_SEGMENTS:
                return True
    return False

def merge_lines(lineA, lineB):
    xA1,yA1,xA2,yA2,slopeA,isPoleA = lineA
    xB1,yB1,xB2,yB2,slopeB,isPoleB = lineB

    pts = np.array([[xA1,yA1],[xA2,yA2],[xB1,yB1],[xB2,yB2]], dtype=np.float32)
    vx,vy,x0,y0 = cv2.fitLine(pts.reshape(-1,1,2),
                              cv2.DIST_L2,0,0.01,0.01).flatten()

    def t(pt):
        return ((pt[0]-x0)*vx + (pt[1]-y0)*vy)/(vx*vx+vy*vy+1e-9)
    tvals = [t(p) for p in pts]
    t0, t1 = min(tvals), max(tvals)
    Ax,Ay = x0 + t0*vx, y0 + t0*vy
    Bx,By = x0 + t1*vx, y0 + t1*vy

    final_slope = math.inf if abs(vx)<1e-9 else float(vy/vx)
    return (float(Ax), float(Ay),
            float(Bx), float(By),
            final_slope,
            bool(isPoleA or isPoleB))


def clamp_line_to_image(Ax, Ay, Bx, By, w, h):
    Ax = max(0, min(Ax, w - 1))
    Ay = max(0, min(Ay, h - 1))
    Bx = max(0, min(Bx, w - 1))
    By = max(0, min(By, h - 1))
    return Ax, Ay, Bx, By


# PCA

def extract_line_from_component(contour, mask_for_points, w, h):
    mask_comp = np.zeros_like(mask_for_points)
    cv2.drawContours(mask_comp, [contour], -1, 255, -1)
    rows, cols = np.where(mask_comp > 0)
    if len(rows) < 2:
        return None

    pts = np.column_stack((cols, rows)).astype(np.float32)
    m = np.mean(pts, axis=0)
    centered = pts - m
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eig(cov)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:,idx]

    ratio = (evals[0]+1e-9)/(evals[1]+1e-9)
    vx,vy = evecs[:,0]
    slope = float('inf') if abs(vx)<1e-9 else float(vy/vx)

    if ratio < PCA_RATIO_THRESHOLD:
        if abs(slope) < 0.5:
            x0,y0,w0,h0 = cv2.boundingRect(contour)
            Ax,Ay = x0, y0+h0/2
            Bx,By = x0+w0, y0+h0/2
            Ax,Ay,Bx,By = clamp_line_to_image(Ax,Ay,Bx,By, w, h)
            return (float(Ax),float(Ay),float(Bx),float(By),0.0,False)
        else:
            return None

    # vertical 
    if abs(slope)>POLE_SLOPE_THRESHOLD or math.isinf(slope):
        median_x = np.median(pts[:,0])
        bw = max(5, (pts[:,0].max()-pts[:,0].min())*VERTICAL_BAND_RATIO)
        band = pts[(pts[:,0]>=median_x-bw)&(pts[:,0]<=median_x+bw)]
        if len(band)>=2:
            pts = band
            m = pts.mean(axis=0)
            centered = pts - m
            cov = np.cov(centered, rowvar=False)
            evals, evecs = np.linalg.eig(cov)
            idx = np.argsort(evals)[::-1]
            vx,vy = evecs[:,idx[0]]
            slope = float('inf') if abs(vx)<1e-9 else float(vy/vx)

    def param_t(px,py):
        return ((px-m[0])*vx + (py-m[1])*vy)/(vx*vx+vy*vy+1e-9)
    tvals = [param_t(px,py) for px,py in pts]

    if not (abs(slope)>POLE_SLOPE_THRESHOLD or math.isinf(slope)):
        xs, ys = pts[:,0], pts[:,1]
        corners = np.array([[xs.min(),ys.min()],
                            [xs.min(),ys.max()],
                            [xs.max(),ys.min()],
                            [xs.max(),ys.max()]], np.float32)
        tb = [param_t(px,py) for px,py in corners]
        t0, t1 = max(min(tvals),min(tb)), min(max(tvals),max(tb))
    else:
        t0, t1 = min(tvals), max(tvals)

    Ax,Ay = m[0]+t0*vx, m[1]+t0*vy
    Bx,By = m[0]+t1*vx, m[1]+t1*vy
    Ax,Ay,Bx,By = clamp_line_to_image(Ax,Ay,Bx,By, w,h)
    isPole = bool(abs(slope)>POLE_SLOPE_THRESHOLD or math.isinf(slope))
    return (float(Ax),float(Ay),float(Bx),float(By),slope,isPole)


# DETECT

def detect_lanes_and_poles(mask):
    cleaned = morphological_cleanup(mask)
    cv2.imshow("Step1: Cleaned Mask", cleaned)
    cv2.waitKey(0)
    cv2.destroyWindow("Step1: Cleaned Mask")

    h, w = cleaned.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    if DEBUG:
        print("[DEBUG] num_labels =", num_labels)

    lines = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        x_,y_,w_,h_ = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP], \
                      stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
        if DEBUG:
            print(f"[DEBUG] Label={lbl}, Area={area}, BBox=({x_},{y_},{w_},{h_})")
        if area < MIN_REGION_AREA:
            continue

        comp_mask = (labels == lbl).astype(np.uint8)
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        lc = max(cnts, key=cv2.contourArea)
        info = extract_line_from_component(lc, cleaned, w, h)
        if info is not None:
            lines.append(info)

   
    # 1) split
    lane_lines = [L for L in lines if not L[-1]]
    pole_lines = [L for L in lines if     L[-1]]
    # 2) estimate VP
    vp = estimate_vanishing_point(lane_lines)
    if vp and DEBUG:
        print(f"[VP] estimated vanishing point at {vp}")
    # 3) keep those lanes pointing within VP_ANG_THRESHOLD
    if vp:
        lane_lines = filter_by_vp(lane_lines, vp, ang_th=VP_ANG_THRESHOLD)

    lines = lane_lines + pole_lines
    # ========================================

    # Merge lines
    merged = True
    while merged:
        merged = False
        new_lines = []
        used = [False]*len(lines)
        for i in range(len(lines)):
            if used[i]: continue
            for j in range(i+1, len(lines)):
                if used[j]: continue
                if should_merge(lines[i], lines[j]):
                    out = merge_lines(lines[i], lines[j])
                    new_lines.append(out)
                    used[i] = used[j] = True
                    merged = True
                    break
            if not used[i]:
                new_lines.append(lines[i])
                used[i] = True
        lines = new_lines


    return lines

# 
# DRAW

g_final_lines = []
g_clicked_lines = []

def draw_lines_on_image(image, lines):
    global g_final_lines
    g_final_lines = lines
    for idx, ln in enumerate(lines):
        x1,y1,x2,y2,slope,isPole = ln
        p1,p2 = (int(x1),int(y1)), (int(x2),int(y2))
        color = (255,0,0) if isPole else (0,0,255)
        cv2.line(image, p1, p2, color, 3)

def distance_point_to_line(px,py, x1,y1,x2,y2):
    L2 = (x2-x1)**2 + (y2-y1)**2
    if L2 < 1e-9:
        return math.hypot(px-x1, py-y1)
    t = ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / L2
    t = max(0, min(1, t))
    projx, projy = x1 + t*(x2-x1), y1 + t*(y2-y1)
    return math.hypot(px-projx, py-projy)

def on_mouse(event, x, y, flags, param):
    global g_final_lines, g_clicked_lines
    if event == cv2.EVENT_LBUTTONDOWN:
        best, idx = float('inf'), -1
        for i, ln in enumerate(g_final_lines):
            d = distance_point_to_line(x,y, *ln[:4])
            if d < best:
                best, idx = d, i
        if idx >= 0 and best < 20:
            x1,y1,x2,y2,_,isPole = g_final_lines[idx]
            rec = {
                "line_index": idx,
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "isPole": bool(isPole)
            }
            g_clicked_lines.append(rec)
            print(f"Clicked line #{idx} => {rec}")
        else:
            print("Click too far from any line.")

def main():
    parser = argparse.ArgumentParser(description='Detect lanes and poles and output endpoints')
    parser.add_argument('--image', required=True, help='Path to the input mask image')
    parser.add_argument('--output', required=True, help='Path for the output JSON endpoints')
    args = parser.parse_args()

    image_path = args.image
    out_path   = args.output

    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading: {image_path}")
        return

    lines = detect_lanes_and_poles(mask)
    display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    draw_lines_on_image(display_img, lines)

    cv2.namedWindow("Detected Lanes & Poles", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Detected Lanes & Poles", on_mouse)

    print("Left-click on lines to record endpoints. Press 'q' to quit.")
    while True:
        cv2.imshow("Detected Lanes & Poles", display_img)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
    cv2.destroyAllWindows()

    with open(out_path, 'w') as f:
        json.dump({"clicked_lines": g_clicked_lines}, f, indent=4)
    print(f"Saved clicked lines => {out_path}")

if __name__ == "__main__":
    main()
