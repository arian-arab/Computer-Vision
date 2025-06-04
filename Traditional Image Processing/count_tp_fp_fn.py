import numpy as np
from score_detection import score_detection

def count_tp_fp_fn(gt,pred,hit_distance,resolution):
    pred = np.array(pred)
    gt = np.array(gt)   
    hit_distance = hit_distance/resolution    
    if gt.shape[0]>0:                
        if pred.shape[0]>0: 
            tp,fp,fn = score_detection(gt, pred, hit_distance) 
        else:
            tp = 0
            fp = 0
            fn = gt.shape[0]
    else:        
        tp = np.nan
        fn = np.nan
        fp = pred.shape[0]       
    return [tp, fp, fn]