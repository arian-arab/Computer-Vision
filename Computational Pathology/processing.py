import numpy as np
import segmentation_models as sm
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import cv2
tf.config.list_physical_devices('GPU')
sm.set_framework('tf.keras')
sm.framework()
plt.rcParams["font.size"] = 32
plt.rcParams["font.family"] = 'Times New Roman'

def plot_img_msk_tils(img, msk, tils):    
    plt.figure()
    plt.imshow(img)
    mask = np.zeros(msk.shape+(3,))
    mask[msk==0] = [255,0,0]
    mask[msk==1] = [0,255,0]
    mask[msk==2] = [0,0,255]
    plt.imshow(mask, alpha = 0.5)
    plt.scatter(tils[:,0], tils[:,1], s = 80, c = 'none', edgecolors = 'k', marker = 's', )  
    plt.axis('off')

def plot_img_msk(img,msk):    
    plt.figure()
    plt.imshow(img)
    mask = np.zeros(msk.shape+(3,))
    mask[msk==0] = [255,0,0]
    mask[msk==1] = [0,255,0]
    mask[msk==2] = [0,0,255]
    plt.imshow(mask, alpha = 0.5)
    plt.axis('off')    

def plot_img_tils(img, tils):    
    plt.figure()
    plt.imshow(img)    
    plt.scatter(tils[:,0], tils[:,1], s = 80, c = 'none', edgecolors = 'k', marker = 's', )  
    plt.axis('off')   
    
def plot_img(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')

def plot_msk(msk):    
    plt.figure()
    mask = np.zeros(msk.shape+(3,))
    mask[msk==0] = [255,0,0]
    mask[msk==1] = [0,255,0]
    mask[msk==2] = [0,0,255]
    plt.imshow(mask)
    plt.axis('off')

############# Load Segmentation and Detection Models ################
seg_model = sm.Unet(backbone_name = 'inceptionv3', input_shape= None, encoder_weights=None, classes = 3, activation = 'Softmax')
seg_model.load_weights('seg_model_color_normalization.hdf5')

det_model = sm.Unet(backbone_name = 'inceptionv3', input_shape= None, encoder_weights = None)
det_model.load_weights('det_model_color_normalization.hdf5')

preprocess_input = sm.get_preprocessing('inceptionv3')

############# Load Img, Msk, and Ref TILs ################  
image = imageio.imread('img.png')
mask = imageio.imread('msk.png')
tils = np.load('tils.npy', allow_pickle=True)

############# Apply Color Normalization on the input Img ################  
from reinhard_color_transfer import reinhard_color_transfer
t_mean = [203.02,134.58,122.55]
t_std = [40.6,5,3.68]
image_norm = reinhard_color_transfer(image,t_mean,t_std)
image_norm = preprocess_input(image_norm)
del t_mean, t_std

############# Run Models on Image ################  
from run_model_on_image import run_model_on_image

one_pass, max_pool, avg_pool = run_model_on_image(model = seg_model, 
                   model_input_shape = (256,256,3), 
                   model_output_shape = (256,256,3),
                   image = image_norm, 
                   stride = (128,128), 
                   blur = True)

one_pass = np.argmax(one_pass, axis = 2)
max_pool = np.argmax(max_pool, axis = 2)
avg_pool = np.argmax(avg_pool, axis = 2)
seg_prediction = np.copy(one_pass)

plot_img_msk(image,one_pass)
plot_img_msk(image,max_pool)
plot_img_msk(image,avg_pool)

one_pass, max_pool, avg_pool = run_model_on_image(model = det_model, 
                   model_input_shape = (256,256,3), 
                   model_output_shape = (256,256,1),
                   image = image_norm, 
                   stride = (128,128), 
                   blur = False)
det_prediction = np.copy(one_pass)

############# Extract Cell Predictions ################
from scipy.spatial.distance import cdist
def extract_predictions(probabilities, confidence_threshold):    
    indices = np.meshgrid(np.arange(0,probabilities.shape[1]),np.arange(0,probabilities.shape[0]))
    indices_x = indices[0]
    indices_y = indices[1]
    indices_x = indices_x.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    indices_y = indices_y.reshape((probabilities.shape[0]*probabilities.shape[1],1))    
    probabilities = probabilities.reshape((probabilities.shape[0]*probabilities.shape[1],1))    
    boxes_pred = np.concatenate((indices_x,indices_y,probabilities),axis = 1)
    boxes_pred = boxes_pred[np.argsort(boxes_pred[:, 2])[::-1]]
    boxes_pred = boxes_pred[boxes_pred[:,2]>=confidence_threshold,:]   
    return boxes_pred

def non_max_supression_distance(points, distance_threshold):
    log_val = np.ones(points.shape[0])
    wanted = []
    for i in range(points.shape[0]):
        if log_val[i]:
            hit = cdist(np.expand_dims(points[i,:2],0),points[:,:2])
            hit = np.argwhere(hit<=distance_threshold)
            log_val[hit] = 0
            wanted.append(points[i,:])
    wanted = np.array(wanted)  
    return wanted
 
def extract_TILs_from_mask(mask, confidence_threshold, distance_threshold):
    temp = extract_predictions(mask, confidence_threshold = confidence_threshold)
    tils = non_max_supression_distance(temp,distance_threshold = distance_threshold)
    return tils

predicted_tils = extract_TILs_from_mask(det_prediction, confidence_threshold=0.1, distance_threshold=8)
plot_img_tils(image, predicted_tils)

no_of_TILs = 0
no_of_stroma = 0
# TILs Score
if len(predicted_tils)>0:
    for i in predicted_tils:
        if seg_prediction[int(i[1]),int(i[0])] == 2:
            no_of_TILs += 1
no_of_stroma += np.count_nonzero(seg_prediction==2) 
tils_score = int(np.round(no_of_TILs*16*16/no_of_stroma*100))

plt.figure()
plt.imshow(image)
mask = np.zeros(seg_prediction.shape+(3,))
mask[seg_prediction==0] = [255,0,0]
mask[seg_prediction==1] = [0,255,0]
mask[seg_prediction==2] = [0,0,255]
plt.imshow(mask, alpha = 0.5)
plt.scatter(predicted_tils[:,0], predicted_tils[:,1], s = 80, c = 'y', marker = 's',alpha = 0.8)
plt.axis('off')
total_stroma = np.sum(seg_prediction==2)
plt.title('Number of TILs = '+ str(predicted_tils.shape[0]) + '\n Number of TILs on Stroma = '+ str(no_of_TILs)+ '\n TILs score = '+ str(tils_score),fontsize =12)