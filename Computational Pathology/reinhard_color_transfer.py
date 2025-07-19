import numpy as np
import cv2

def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std

def reinhard_color_transfer(s,t_mean,t_std):
    s = cv2.cvtColor(s,cv2.COLOR_BGR2LAB)    
    s_mean, s_std = get_mean_and_std(s)    
    transfer = np.zeros(s.shape)
    transfer = ((s-s_mean)*(t_std/s_std))+t_mean
    transfer = np.round(transfer)
    transfer[transfer<0] = 0
    transfer[transfer>255] = 255
    transfer = transfer.astype('uint8')
    transfer = cv2.cvtColor(transfer,cv2.COLOR_LAB2BGR)
    return transfer