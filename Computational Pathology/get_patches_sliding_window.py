import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

from utils.get_level import get_level

def get_patches_sliding_window(data, mpp, patch_w = 256, patch_h = 256, stride_x = 256, stride_y = 256, include_boundary = True, include_tissue_mask = True, mask_threshold = 30, method = 'pyramid'):
    # returns all patches using sliding window technique, follows the tissue mask if provided
    # the tissue mask should contain only 0s and 1s
    if mpp == 0:
        print('mpp can not be zero, it will change to the slide mpp')
        mpp = data['slide_mpp']
    
    w_0 = data['slide_w'] # w_0 slide width at the highest magnification
    h_0 = data['slide_h'] # h_0 slide height at the highest magnification        
    
    if method == 'level_zero':    
    
        scale_factor = mpp / data['slide_mpp']
    
        if include_tissue_mask:
            tissue_mask = data['tissue_mask']        
            tissue_mask[tissue_mask!=0] = 1            
            tissue_mask = cv2.resize(tissue_mask, (w_0, h_0), interpolation=cv2.INTER_NEAREST) # generates tissue mask at level 0        
    
        patch_w_ = int(patch_w * scale_factor)
        patch_h_ = int(patch_h * scale_factor)
        stride_x_ = int(stride_x * scale_factor)
        stride_y_ = int(stride_y * scale_factor)
        
        x_vals = np.arange(0, w_0, stride_x_) # x values along the x-axis at level 0
        y_vals = np.arange(0, h_0, stride_y_) # y values along the y-axis at level 0
        
        patches_images = []
        patches_coordinates = []    
          
        for y in tqdm(y_vals):
            for x in x_vals:
                
                if include_boundary: # includes patches at the boundaries of the slide image
                    
                    if x + patch_w_ > w_0:   
                        x = w_0 - patch_w_ # to ensure boundary patches are included                
                   
                    if y + patch_h_ > h_0:
                        y = h_0 - patch_h_ # to ensure boundary patches are included 
                    
                    if include_tissue_mask:
                        patch_mask = tissue_mask[y : y + patch_h_, x : x + patch_w_]  
                        patch_mask_area = 100 * np.sum(patch_mask) / (patch_mask.shape[0] * patch_mask.shape[1])
                        
                        if patch_mask_area >= mask_threshold: # only extract patch if matches the threshold value                                       
                            region = data['slide'].read_region((x, y), level = 0, size=(patch_w_, patch_h_))                        
                            region = region.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
                            patches_images.append(np.array(region.convert("RGB")))
                            patches_coordinates.append(np.array([x,y])) 
                    
                    else:
                        region = data['slide'].read_region((x, y), level = 0, size=(patch_w_, patch_h_))                   
                        region = region.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
                        patches_images.append(np.array(region.convert("RGB")))
                        patches_coordinates.append(np.array([x,y]))                         
                    
                else:
                   
                    if x + patch_w_ <= w_0 and y + patch_h_ <= h_0:     
                        
                        if include_tissue_mask:
                            patch_mask = tissue_mask[y : y + patch_h_, x : x + patch_w_]  
                            patch_mask_area = 100 * np.sum(patch_mask) / (patch_mask.shape[0] * patch_mask.shape[1])
                            
                            if patch_mask_area >= mask_threshold: # only extract patch if matches the threshold value                           
                                region = data['slide'].read_region((x, y), level = 0, size=(patch_w_, patch_h_))                            
                                region = region.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
                                patches_images.append(np.array(region.convert("RGB")))
                                patches_coordinates.append(np.array([x,y])) 
                        
                        else:
                            region = data['slide'].read_region((x, y), level = 0, size=(patch_w_, patch_h_))                        
                            region = region.resize((patch_w, patch_h), Image.Resampling.LANCZOS)                        
                            patches_images.append(np.array(region.convert("RGB")))
                            patches_coordinates.append(np.array([x,y]))  
                            
    if method == 'pyramid':
        best_level = get_level(data, mpp)
        best_level_mpp = data['slide_mpps'][best_level] 

        if include_tissue_mask:
            tissue_mask = data['tissue_mask']        
            tissue_mask[tissue_mask!=0] = 1            
            tissue_mask = cv2.resize(tissue_mask, (w_0, h_0), interpolation=cv2.INTER_NEAREST) # generates tissue mask at level 0
        
        patch_w_ = int(patch_w * (mpp / best_level_mpp))
        patch_h_ = int(patch_h * (mpp / best_level_mpp))
        
        patch_w__ = int(patch_w * (mpp / data['slide_mpp']))
        patch_h__ = int(patch_h * (mpp / data['slide_mpp']))
        
        stride_x_ = int(stride_x * (mpp / data['slide_mpp']))
        stride_y_ = int(stride_y * (mpp / data['slide_mpp']))
        
        x_vals = np.arange(0, w_0, stride_x_) # x values along the x-axis at level 0
        y_vals = np.arange(0, h_0, stride_y_) # y values along the y-axis at level 0
        
        patches_images = []
        patches_coordinates = []    
          
        for y in tqdm(y_vals):
            for x in x_vals:
                
                if include_boundary: # includes patches at the boundaries of the slide image
                    
                    if x + patch_w__ > w_0:   
                        x = w_0 - patch_w__ # to ensure boundary patches are included                
                   
                    if y + patch_h__ > h_0:
                        y = h_0 - patch_h__ # to ensure boundary patches are included 
                    
                    if include_tissue_mask:
                        patch_mask = tissue_mask[y : y + patch_h__, x : x + patch_w__]
                        patch_mask_area = 100 * np.sum(patch_mask) / (patch_mask.shape[0] * patch_mask.shape[1])
                        
                        if patch_mask_area >= mask_threshold: # only extract patch if matches the threshold value                                
                            region = data['slide'].read_region((x, y), level = best_level, size=(patch_w_, patch_h_))                        
                            region = region.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
                            patches_images.append(np.array(region.convert("RGB")))
                            patches_coordinates.append(np.array([x,y])) 
                    
                    else:
                        region = data['slide'].read_region((x, y), level = best_level, size=(patch_w_, patch_h_))                   
                        region = region.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
                        patches_images.append(np.array(region.convert("RGB")))
                        patches_coordinates.append(np.array([x,y]))                         
                    
                else:
                   
                    if x + patch_w__ <= w_0 and y + patch_h__ <= h_0:     
                        
                        if include_tissue_mask:
                            patch_mask = tissue_mask[y : y + patch_h__, x : x + patch_w__]
                            patch_mask_area = 100 * np.sum(patch_mask) / (patch_mask.shape[0] * patch_mask.shape[1])
                            
                            if patch_mask_area >= mask_threshold: # only extract patch if matches the threshold value                        
                                region = data['slide'].read_region((x, y), level = best_level, size=(patch_w_, patch_h_))                            
                                region = region.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
                                patches_images.append(np.array(region.convert("RGB")))
                                patches_coordinates.append(np.array([x,y])) 
                        
                        else:
                            region = data['slide'].read_region((x, y), level = best_level, size=(patch_w_, patch_h_))                        
                            region = region.resize((patch_w, patch_h), Image.Resampling.LANCZOS)                        
                            patches_images.append(np.array(region.convert("RGB")))
                            patches_coordinates.append(np.array([x,y]))   
                            
             
    print('Number of Extracted Patches = ' + str(len(patches_images)))    
    data['patches'] = patches_images
    data['patches_coordinates'] = patches_coordinates
    data['patches_mpp'] = mpp
    
    data['patches_features'] = None    
    data['patches_features_method'] = None
    
    data['patches_sampled'] = None
    data['patches_sampled_coordinates'] = None
    data['patches_sampled_method'] = None
    return data