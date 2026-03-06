import openslide
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pad_image(image, target_size):
    h, w, c = image.shape
    new_h = ((h + target_size - 1) // target_size) * target_size
    new_w = ((w + target_size - 1) // target_size) * target_size

    pad_h = new_h - h
    pad_w = new_w - w    
    
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='wrap')
    
    return padded.astype(np.uint8)


def extract_patches_from_image_sliding_window(image, patch_size, stride, return_coordinates = False):
    
    if len(image.shape) == 3:
        h, w, _  = image.shape
    elif len(image.shape) == 2:
        h, w  = image.shape
    else:
        print('input image should a 2- or 3- dimensional NumPy array')
        return []
    
    if patch_size <= 0:
        patch_size = min(h, w)
        print(f'patch size was smaller than zero, changed to min(h, w) = {patch_size}')
    
    if stride <= 0:
        stride = min(h, w)  
        print(f'stride was smaller than zero, changed to min(h, w) = {stride}')
    
    patch_size_x = patch_size
    patch_size_y = patch_size
    
    stride_x = stride    
    stride_y = stride    
    
    if patch_size_y > h:
        patch_size_y = h
        stride_y = h
        
    if patch_size_x > w:
        patch_size_x = w
        stride_x = w

    x = np.arange(0, w, stride_x)
    y = np.arange(0, h, stride_y)
    
    x = [i for i in x if i + patch_size_x <= w]
    y = [i for i in y if i + patch_size_y <= h]
    
    x.append(w - patch_size_x)
    y.append(h - patch_size_y)
    
    x_vals = np.unique(x)
    y_vals = np.unique(y)
    
    patches = []
    for y in y_vals:
        for x in x_vals:
            patches.append(image[y : y + patch_size_y, x : x + patch_size_x])
    
    if return_coordinates:
        return patches, x_vals, y_vals
    else:
        return patches


def stitch_predictions(image, preds, x_vals, y_vals):
    
    n_channels, patch_size, patch_size = preds[0].shape 
    
    stitched_image = np.zeros((n_channels, image.shape[0], image.shape[1]))
    
    counter = np.zeros((image.shape[0], image.shape[1]))
          
    idx = 0
    for y in y_vals:
        for x in x_vals:
            stitched_image[:, y : y + patch_size, x: x + patch_size] += preds[idx]
            counter[y : y + patch_size, x: x + patch_size] += 1            
            idx += 1
            
    counter[counter == 0] = 1
    
    stitched_image /= counter 
    
    return stitched_image


def get_tissue_mask_otsu(thumbnai):
    
    from skimage import color    
    from skimage.filters import gaussian
    from skimage import filters, morphology
    
    gray = color.rgb2gray(thumbnail)  
    gray_blurred = gaussian(gray, sigma=1)
    
    # Otsu thresholding    
    otsu_thresh = filters.threshold_otsu(gray_blurred)
    binary_mask = gray_blurred < otsu_thresh  # Tissue usually darker than background
    
    # Clean up using morphological operations, remove small holes and specks    
    cleaned_mask = morphology.remove_small_holes(binary_mask, area_threshold=500)
    cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=100)
    tissue_mask = cleaned_mask.astype('uint8') 
    
    return tissue_mask



def jpeg_compression_transform(image, quality=80):
    """
    Apply JPEG compression to an input PIL image without OpenCV.
    """
    from PIL import Image
    from io import BytesIO

    # Save image to a JPEG buffer with specified quality
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Reload the compressed image
    image = Image.open(buffer)

    # Resize to 512x512
    image = image.resize((512, 512), Image.BILINEAR)

    return image


# import requests

# url = "https://zenodo.org/records/14507273/files/Tissue_Detection_MPP10.pth?download=1"
# output_path = "./Tissue_Detection_MPP10.pth"

# response = requests.get(url, stream=True)
# response.raise_for_status()

# with open(output_path, "wb") as f:
#     for chunk in response.iter_content(chunk_size=8192):
#         if chunk:
#             f.write(chunk)

# print(f"Model downloaded and saved to {output_path}")







def get_tissue_mask_grandqc(thumbnail, mpp):
    thumbnail_padded = pad_image(thumbnail, target_size = 512)

    patch_size = 512
    stride = 256
    patches, x_vals, y_vals = extract_patches_from_image_sliding_window(thumbnail_padded, 
                                                                        patch_size, 
                                                                        stride, 
                                                                        return_coordinates = True)      
    
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    import segmentation_models_pytorch as smp        
    weights_path = './Tissue_Detection_MPP10.pth'       


    model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b0', encoder_weights=None, weights_only=True, classes=2, activation=None)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    from torchvision import transforms
    eval_transforms = transforms.Compose([                       
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    batch_size = 32
    
    predictions = []
    with torch.no_grad():
        from tqdm import tqdm
        for i in tqdm(range(0, len(patches), batch_size)):
            batch = patches[i:i+batch_size]
                       
            batch_tensors = [eval_transforms(img) for img in batch]            
            batch_tensors = torch.stack(batch_tensors) 
            batch_tensors = batch_tensors.to(DEVICE)

            preds = model(batch_tensors)  
            preds_np = preds.cpu().numpy()
            predictions.extend(list(preds_np))
            
            
    pred_sw = stitch_predictions(thumbnail_padded, predictions, x_vals, y_vals)

    pred_sw = pred_sw[:, :thumbnail.shape[0], :thumbnail.shape[1]]    

    pred_sw = np.argmax(pred_sw, axis = 0)   

    tissue_mask = (pred_sw).astype('uint8')
    
    return tissue_mask
    
    
    
full_path = './beetle.tif'

slide = openslide.OpenSlide(full_path)     
slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
slide_mpps = np.array([i * slide_mpp for i in slide.level_downsamples])

mpp = 10 # model is trained at 10 mpp (1X) 
thumbnail_reduction_factor = mpp / slide_mpp 
thumbnail = slide.get_thumbnail((slide_w // thumbnail_reduction_factor, 
     
                                    slide_h // thumbnail_reduction_factor))        
thumbnail = np.array(thumbnail.convert("RGB"))
plt.figure()
plt.imshow(thumbnail)
plt.axis('off')
plt.show()


tissue_mask = get_tissue_mask_grandqc(thumbnail, mpp)
 
rgba_mask = np.zeros((*tissue_mask.shape, 4))
rgba_mask[tissue_mask == 1] = [0, 1, 0, 0.3]  # Green with alpha 0.3
rgba_mask[tissue_mask == 0] = [0, 0, 0, 0.0]  # Fully transparent   
fig, ax = plt.subplots()
ax.imshow(thumbnail) 
ax.imshow(rgba_mask)  
ax.axis('off')                   
plt.show()



tissue_mask = get_tissue_mask_otsu(thumbnail)

rgba_mask = np.zeros((*tissue_mask.shape, 4))
rgba_mask[tissue_mask == 1] = [0, 1, 0, 0.3]  # Green with alpha 0.3
rgba_mask[tissue_mask == 0] = [0, 0, 0, 0.0]  # Fully transparent   
fig, ax = plt.subplots()
ax.imshow(thumbnail) 
ax.imshow(rgba_mask)  
ax.axis('off')                   
plt.show()