import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def refine_mask_with_crf(semantic_mask, image_path, sxy_val=20, srgb_val=10, compat_val=5, use_vehicles=False):
    """
    Refines a general pixel mask (including vehicles optionally) 
    by snapping probabilities to the edges of objects in the raw image.
    """
    final_shape = [semantic_mask.shape[0], semantic_mask.shape[1], 2]
    final_mask = np.zeros(final_shape)

    for i in range(semantic_mask.shape[0]):
        for j in range(semantic_mask.shape[1]): 
            # Focus on road or other object probabilities from model
            final_mask[i,j,1] = semantic_mask[i,j,5]
            if use_vehicles:
                # Include vehicles if requested
                final_mask[i,j,1] += semantic_mask[i,j,13] + semantic_mask[i,j,11]
            final_mask[i,j,0] = 1 - final_mask[i,j,1]
            
    final_mask = np.moveaxis(final_mask, -1, 0)  
    semantic_mask = np.moveaxis(semantic_mask, -1, 0)

    U = unary_from_softmax(final_mask) 
    img = cv2.imread(image_path)  

    img = img.copy(order='C')
    U = U.copy(order='C').astype(np.float32) 

    crf = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
    crf.setUnaryEnergy(U)
    crf.addPairwiseBilateral(sxy=sxy_val, srgb=srgb_val, rgbim=img, compat=compat_val) 
    
    Q = crf.inference(5)
    refined_map = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1])) 
    
    return refined_map


def refine_road_mask_with_crf(semantic_mask, image_path, sxy_val=100, srgb_val=10, compat_val=5):
    """
    Specifically refines just the road area from the semantic mask,
    isolating the drivable plane probabilities.
    """
    final_shape = [semantic_mask.shape[0], semantic_mask.shape[1], 2]
    final_mask = np.zeros(final_shape)

    for i in range(semantic_mask.shape[0]):
        for j in range(semantic_mask.shape[1]): 
            # Channel 1 is set from the logical 'road' class
            final_mask[i,j,1] = semantic_mask[i,j,0]
            final_mask[i,j,0] = 1 - final_mask[i,j,1]
            
    final_mask = np.moveaxis(final_mask, -1, 0)  
    semantic_mask = np.moveaxis(semantic_mask, -1, 0)

    U = unary_from_softmax(final_mask) 
    img = cv2.imread(image_path)  

    img = img.copy(order='C')
    U = U.copy(order='C').astype(np.float32) 

    crf = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
    crf.setUnaryEnergy(U)
    crf.addPairwiseBilateral(sxy=sxy_val, srgb=srgb_val, rgbim=img, compat=compat_val)
    
    Q = crf.inference(5)
    refined_map = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1])) 
    
    return refined_map
