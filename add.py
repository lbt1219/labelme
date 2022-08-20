import cv2
import numpy as np

#img1 = cv2.imread('/Volumes/My_Passport/LBT/DATASET/anti-vibration_hammer/timdatasets/result/实例分割/分割/2-1.png' , -1)

mask = cv2.imread('/Volumes/My_Passport/LBT/DATASET/anti-vibration_hammer/timdatasets/result/实例分割/分割/5.png' , -1)

#print(img1.shape,img2.shape)

#mask = cv2.add(img1,img2)

#cv2.imwrite('2add.jpg',mask)

black_pixels = np.where(
    (mask[:, :, 0] == 0) &
    (mask[:, :, 1] == 0) &
    (mask[:, :, 2] == 0)
)

# set those pixels to white
mask[black_pixels] = [255, 255, 255]

cv2.imwrite('5dst.jpg',mask)