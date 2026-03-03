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


class CLAHE(object):
    # histogram equalisation
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        HSV[:, :, 0] = self.clahe.apply(HSV[:, :, 0])
        img = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
        return img            


class UNet_down_block(torch.nn.Module):
    
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.InstanceNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.InstanceNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet_up_block(torch.nn.Module):
    
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.InstanceNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.InstanceNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout2d(p=0.2)

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, self.dropout(prev_feature_map)), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(torch.nn.Module):
    
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, True)
        self.down_block7 = UNet_down_block(512, 1024, True)

        self.mid_conv1 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = torch.nn.InstanceNorm2d(1024)
        self.mid_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = torch.nn.InstanceNorm2d(1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.InstanceNorm2d(16)
        self.last_conv2 = torch.nn.Conv2d(16, 2, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.down_block6(x5)
        x7 = self.down_block7(x6)
        x7 = self.relu(self.bn1(self.mid_conv1(x7)))
        x7 = self.relu(self.bn2(self.mid_conv2(x7)))
        x = self.up_block1(x6, x7)
        x = self.up_block2(x5, x)
        x = self.up_block3(x4, x)
        x = self.up_block4(x3, x)
        x = self.up_block5(x2, x)
        x = self.up_block6(x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x    


def get_tissue_mask_grandqc(thumbnail, mpp):
    thumbnail_padded = pad_image(thumbnail, target_size = 512)

    patch_size = 512
    stride = 256
    patches, x_vals, y_vals = extract_patches_from_image_sliding_window(thumbnail_padded, 
                                                                        patch_size, 
                                                                        stride, 
                                                                        return_coordinates = True)  
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    weights_path = './path_profiler.pth'        
    batch_size = 32

    checkpoint = torch.load(weights_path, map_location=DEVICE)        
    state_dict = checkpoint['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove 'module.' prefix
        new_state_dict[name] = v

    model = UNet().to(DEVICE)
    model.load_state_dict(new_state_dict)
    model.eval()
     
    from torchvision import transforms
    eval_transforms = transforms.Compose([
        CLAHE(),  # apply histogram equalization in HSV space            
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1] range
        ])


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