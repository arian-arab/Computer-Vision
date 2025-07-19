import numpy as np
import matplotlib.pyplot as plt
import cv2

def run_model_on_image(model, model_input_shape, model_output_shape, image, stride, blur = True):    
    blur_size = 50
    stride_x, stride_y = stride
    
    ########### chekc if all input dimensions are correct ##########
    image_shape = image.shape
    
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    if len(image_shape) != 3:
        raise ValueError('Input image should be 3-dimensional')     
    
    image_y = image_shape[0]
    image_x = image_shape[1]
    image_c = image_shape[2]   
    
    if len(model_input_shape) != 3:
        raise ValueError('Model input should be 3-dimensional') 
        
    model_y_in = model_input_shape[0]
    model_x_in = model_input_shape[1]
    model_c_in = model_input_shape[2] 
    
    if len(model_output_shape) != 3:
        raise ValueError('Model output should be 3-dimensional') 
    
    model_y_out = model_output_shape[0]
    model_x_out = model_output_shape[1]
    model_c_out = model_output_shape[2]  
    
    if model_y_in != model_x_in:
        raise ValueError('Model input shapes are not equal (x, y)')   
    
    if model_y_out != model_x_out:
        raise ValueError('Model output shapes are not equal (x, y)')                   
    
    if image_c != model_c_in:
        raise ValueError('Image input shape is different with model input shape')
    
    ############### create padded image ###########    
    if image_y == model_y_in:
        padded_image_y = model_y_in
    else:
        padded_image_y = ((image_y // model_y_in)+1)*model_y_in
    
    if image_x == model_x_in:
        padded_image_x = model_x_in
    else:
        padded_image_x = ((image_x // model_x_in)+1)*model_x_in
        

    padded_image = np.zeros((padded_image_y, padded_image_x, image_c))
    padded_image[:image_y, :image_x, :] = image
        
    ############# One Pass to the Model - Fully Convolutional ###########    
    one_pass_prediciton = model.predict(np.expand_dims(padded_image, axis = 0))
    one_pass_prediciton = one_pass_prediciton[0][:image_y, :image_x,:]
    if blur:
        one_pass_prediciton = cv2.blur(one_pass_prediciton, (blur_size, blur_size))
            
    plt.figure()
    plt.imshow(one_pass_prediciton)
    plt.axis('off')
    plt.show()
    
        
    ############# Sliding Window Technique ###########
    y = np.arange(0,padded_image_y,stride_y)
    x = np.arange(0,padded_image_x,stride_x)    
    
    y = [i for i in y if i+model_y_in <= padded_image_y]
    x = [i for i in x if i+model_x_in <= padded_image_x]
    
    patches = []
    for j in y:
        for i in x:
            patches.append(padded_image[j:j+model_y_in, i:i+model_x_in, :])    
    patches = np.array(patches)     
    
    patches_predictions = model.predict(patches, verbose = 1, batch_size = 32)
    
    ############# Max Pooling Method ###############
    max_pool_prediciton = np.zeros((padded_image_y,padded_image_x,model_c_out))    
    idx = 0
    for j in y:
        for i in x:
            max_pool_prediciton[j:j+model_y_in, i:i+model_x_in,:] = np.maximum(max_pool_prediciton[j:j+model_y_in, i:i+model_x_in,:], patches_predictions[idx])
            idx += 1
    max_pool_prediciton = max_pool_prediciton[:image_y, :image_x, :]    
    if blur:
        max_pool_prediciton = cv2.blur(max_pool_prediciton, (blur_size, blur_size))
    
    plt.figure()
    plt.imshow(max_pool_prediciton)
    plt.axis('off')
    plt.show()    
     
    ############# Avg Pooling Method ###############      
    avg_pool_prediciton = np.zeros((padded_image_y,padded_image_x,model_c_out))
    counter = np.zeros((padded_image_y,padded_image_x))
    idx = 0
    for j in y:
        for i in x:
            avg_pool_prediciton[j:j+model_y_in, i:i+model_x_in,:] = np.add(avg_pool_prediciton[j:j+model_y_in, i:i+model_x_in,:], patches_predictions[idx])
            counter[j:j+model_y_in, i:i+model_x_in] += 1
            idx += 1
    avg_pool_prediciton = avg_pool_prediciton / np.max(counter)
    avg_pool_prediciton = avg_pool_prediciton[:image_y, :image_x, :]
    if blur:
        avg_pool_prediciton = cv2.blur(avg_pool_prediciton, (blur_size, blur_size))
    
    plt.figure()
    plt.imshow(avg_pool_prediciton)
    plt.axis('off')
    plt.show()   
    
    return one_pass_prediciton, max_pool_prediciton, avg_pool_prediciton
    
