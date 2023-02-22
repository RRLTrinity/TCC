import numpy as np
import os 
import scipy
import skimage
from skimage import data,io
import cv2 
import matplotlib.pyplot as plt
import time
import mediapipe
import sys
import pdb
import face_detect


def gaussianPyrDown (image,levels):
    cI = image.copy()
    gp = [image]
    #print('Tamanho original',len(image[0]))
    for i in range(0,levels):
        cI = cv2.pyrDown(cI)     
        #print('nivel', i + 1,':', len(cI[0]))
        gp.append(cI)
    return gp

def gaussianPyrUp (gp,levels,image):
    cI = image.copy()
    gpu = [image]

    for i in range(0,levels):
        cI = cv2.pyrUp(cI)
        ccI = cI[0:len(gp[levels-i-1]),0:len(gp[levels-i-1][0])]
        #print('nivel',(levels-i +1),':',len(ccI[0]),'-',len(gp[levels-i-1][0]))
        gpu.append(ccI)
    return gpu

def idealPassBandFilter(image,freqs):
    fft_image = np.fft.fftshift(np.fft.fft2(image))

    sos = scipy.signal.butter(1,[freqs[0],freqs[1]], 'bandpass', fs=30, output='sos')
    fft_image_filtered = scipy.signal.sosfilt(sos, fft_image)

    filtered_image = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image_filtered)))

    return filtered_image

def colorMagnification(image,alpha,attenuation):
    image_m = image*alpha
    image_m[:,:,1]*=attenuation
    image_m[:,:,2]*=attenuation
    return image_m

def eulerianvideomagnification(image,levels,alpha,attenuation):
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rgb_float = skimage.img_as_float(rgb)
    YIQ = skimage.color.rgb2yiq(rgb)

    gp = gaussianPyrDown(YIQ,levels)

    ifft_image = idealPassBandFilter(gp[-1],[0.8333,1])    
    ifft_image = colorMagnification(ifft_image,alpha,attenuation)

    gpu = gaussianPyrUp(gp,levels,ifft_image)
    image_reconstruct = gpu[-1]
    new_rgb1 = skimage.color.yiq2rgb(image_reconstruct)
    pdi2 = new_rgb1[:,:,::-1]
    final_image = cv2.add(image_reconstruct,YIQ)
    new_rgb = skimage.color.yiq2rgb(final_image)
    pdi = new_rgb[:,:,::-1]

    return pdi    

image = cv2.imread("teste2.png",cv2.IMREAD_COLOR)

levels = 4
alpha = 50
attenuation = 1

my_test = eulerianvideomagnification(image,levels,alpha,attenuation)




vid = cv2.VideoCapture("modelo1treino.mp4")

while(1):
  _, image = vid.read()
  roi = face_detect.roi(image)
  pdi = eulerianvideomagnification(roi,levels,alpha,attenuation)
  #image = face_detect.facedetection(image)
  #roi = face_detect.roi(image)
    
  cv2.imshow("imagem", image)
  cv2.imshow("pdi",pdi)

  cv2.waitKey(5)
cv2.destroyAllWindows()


