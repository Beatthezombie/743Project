import numpy as np
import cv2


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use red green blue to represent different classes

def visualize_pred(windowname, pred, gt, direct):
    
    pred = np.transpose(pred * 255, (1,2,0)).astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    gt = np.transpose(gt * 255, (1,2,0)).astype(np.uint8)
    gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
    direct = np.transpose(direct * 255, (1,2,0)).astype(np.uint8)
    direct = cv2.cvtColor(direct, cv2.COLOR_RGB2BGR)

    h,w,_ = gt.shape
    image = np.zeros([h,w*3,3], np.uint8)
    image[:h,w:2 * w] = pred
    image[:h,2 * w:] = gt
    image[:h,:w] = direct
    cv2.imwrite(windowname+".png", image)