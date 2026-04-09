import numpy as np
import cv2
import argparse
from ai_segmentation import process_image_with_ai
from crf_smoother import refine_mask_with_crf, refine_road_mask_with_crf

def extract_lane_mask(img, semantic_mask): 
    """
    Sub-routine to deduce strict lane coordinates from the smoothed image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (semantic_mask == 1)
    gray[semantic_mask != 1] = 0 

    mean = np.sum(gray.flatten()) / np.sum(mask.flatten()) 
    std = np.sqrt(np.sum((gray[semantic_mask == 1] - mean)**2) / np.sum(mask.flatten())) 

    gray = gray.astype('float32')
    gray[mask] = (gray[mask] - mean) / std * 20 + 125
    gray = np.clip(gray, 0, 255).astype('uint8') 
     
    ret, thresh = cv2.threshold(gray, 170, 256, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    lane_map = np.zeros(gray.shape) 
    cv2.drawContours(lane_map, contours, -1, 255 , -1)
    
    return lane_map


def build_final_mask(img_name, outputfile, use_car=False):
    """
    Main pipeline joining AI execution with CRF edge-smoothing
    to create a binary map of lane boundaries.
    """
    img = cv2.imread(img_name)
    
    print(f"[*] Processing image: {img_name}")
    print("[*] Running DeepLab AI (this may take a moment)...")
    
    (seg_map, seg_map_result) = process_image_with_ai(img_name)

    print("[*] Smoothing mask edges natively with Dense CRF...")
    crf_mask = refine_mask_with_crf(seg_map, img_name, sxy_val=20, srgb_val=10, compat_val=5, use_vehicles=use_car)
    crf_mask_road = refine_road_mask_with_crf(seg_map, img_name, sxy_val=100, srgb_val=10, compat_val=5) 

    print("[*] Extracting inner lanes...")
    lane_mask = extract_lane_mask(img, crf_mask_road) 

    print("[*] Blending final results...")
    merged_mask = np.zeros(lane_mask.shape) 

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):  
            if crf_mask[i,j] == 1:
                merged_mask[i,j] = 255
            elif crf_mask_road[i,j] == 1:
                if lane_mask[i,j] > 10:
                    merged_mask[i,j] = 255
                    
    print(f"[*] Done! Saving lane mask to: {outputfile}")
    cv2.imwrite(outputfile, merged_mask) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate lane mask using AI and CRF for thesis research.')
    parser.add_argument('-i', '--input', required=True, help='Path to input image')
    parser.add_argument('-o', '--output', required=True, help='Path to output mask image')
    parser.add_argument('-v', '--vehicles', action='store_true', help='Set flag to include vehicle filtering')
    
    args = parser.parse_args()
    build_final_mask(args.input, args.output, args.vehicles)
