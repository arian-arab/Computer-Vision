import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as patches

def plot_data(data, save_plot = False): 
    fig, ax = plt.subplots()
    
    thumbnail = data['thumbnail']
    tissue_mask = data['tissue_mask']
    
    thumbnail = cv2.resize(thumbnail, (tissue_mask.shape[1],
                                       tissue_mask.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
    ax.imshow(thumbnail) 
    
    rgba_mask = np.zeros((*tissue_mask.shape, 4))
    rgba_mask[tissue_mask == 1] = [0, 1, 0, 0.4]  # Green with alpha 0.4
    rgba_mask[tissue_mask == 0] = [0, 0, 0, 0.0]  # Fully transparent   
    ax.imshow(rgba_mask)  
    ax.axis('off')                   
        
    if data['patches'] is not None and len(data['patches']) > 0 :
        if len(data['patches']) < 5000: # grid window if number of patches < 5000   
            
            patches_images = data['patches']
            patches_coordinates = data['patches_coordinates']     
            patches_mpp = data['patches_mpp']

            w_0 = data['slide_w'] # slide width at the highest magnification       
            
            w_t = tissue_mask.shape[1]
            
            tissue_mask_reduction_factor = w_0 / w_t
            
            patches_reduction_factor = patches_mpp / data['slide_mpp']

            patch_w = patches_images[0].shape[1] 
            patch_h = patches_images[0].shape[0]                
            patch_w *=  patches_reduction_factor / tissue_mask_reduction_factor
            patch_h *=  patches_reduction_factor / tissue_mask_reduction_factor
            
            patches_coordinates_ = [(i[0] / tissue_mask_reduction_factor , i[1] / tissue_mask_reduction_factor ) for i in patches_coordinates]            
            
            for x, y in patches_coordinates_:            
                square = patches.Rectangle((x, y), patch_w, patch_h, fill = False, edgecolor = 'red', linewidth = 0.5)
                ax.add_patch(square)
            ax.set_title('# Patches = ' + str(len(patches_images)))                  
        else:
            print('number of patches are more than 5000, hence did not plotted')
                   
            
    if data['patches_sampled'] is not None and len(data['patches_sampled']) > 0:
        if len(data['patches_sampled']) < 5000: # grid window if number of patches sampled < 5000   
            
            patches_images = data['patches']
            patches_coordinates = data['patches_coordinates']
            patches_sampled = data['patches_sampled']
            patches_sampled_coordinates = data['patches_sampled_coordinates']    
            patches_mpp = data['patches_mpp']
            
            w_0 = data['slide_w'] # slide width at the highest magnification  
            
            patches_reduction_factor = patches_mpp / data['slide_mpp']
            
            w_t = tissue_mask.shape[1]
            
            tissue_mask_reduction_factor = w_0 / w_t    

            patch_w = patches_images[0].shape[1]
            patch_h = patches_images[0].shape[0]                
            patch_w *=  patches_reduction_factor / tissue_mask_reduction_factor 
            patch_h *=  patches_reduction_factor / tissue_mask_reduction_factor
            
            patches_coordinates_ = [(i[0] / tissue_mask_reduction_factor , i[1] / tissue_mask_reduction_factor ) for i in patches_sampled_coordinates]
            
            for x, y in patches_coordinates_:            
                square = patches.Rectangle((x, y), patch_w, patch_h, fill = False, edgecolor = 'blue', linewidth = 0.8)
                ax.add_patch(square)    
            ax.set_title('# Patches = ' + str(len(patches_images)) + '\n' + '# Sampled Patches = ' + str(len(patches_sampled)))                                      
        else:
            print('number of patches are more than 5000, hence did not plotted')
            
    if save_plot:        
        slide_name = data['slide_name'].split('.')[0]
        save_path = data['save_path'] + slide_name + '_thumbnail_tissue_mask.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close(fig)         
    else:
        plt.show()   