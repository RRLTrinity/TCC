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


#Carregando imagem (homem aranha)


cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

 
image = cv2.imread("rosto_1.jpg",cv2.IMREAD_COLOR)
image2 = cv2.imread("rosto_1.jpg",cv2.IMREAD_COLOR)
#image = cv2.imread("Gama.jpg",cv2.IMREAD_COLOR)



#Exibindo imagem

#cv2.imshow("CV Galáxia",image) # Exibe imagem
#cv2.waitKey(0) # Mantém janela aberta até ser fechada
#cv2.destroyAllWindows()

# Convertendo de BGR para outro formato
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Criando parametros para imagem (escalas de vermelho)

red_lower_range = np.array([0,100,100])
red_upper_range = np.array([5,255,255])

# Criando parametros para imagem (escalas de vermelho)

#blue_lower_range = np.array([110,50,50])
#blue_upper_range = np.array([130,255,255])
blue_lower_range = np.array([95,50,50])
blue_upper_range = np.array([130,255,255])

#criando parametros para imagem (escalas de verde)


green_lower_range = np.array([35,20,20])
green_upper_range = np.array([75,255,255])

# Criando máscara

#vermelho
#mask = cv2.inRange(hsv,red_lower_range,red_upper_range)
#azul
#mask = cv2.inRange(hsv,blue_lower_range,blue_upper_range)
#amarelo
mask = cv2.inRange(hsv,red_lower_range,red_upper_range)



#comparando as imagens

#cv2.imshow('image_window_name', image)
#cv2.imshow('mask_window_name', mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Detectando rostos na imagem
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)



# Desenhando retangulos ao redors de rostos
for (x, y, w, h) in faces:
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    ROI= np.array([[(x+0.25*w,y),(x+0.8*w,y),(x+0.8*w,y+0.2*h),(x+0.25*w,y+0.2*h)]], dtype= np.int32)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blank= np.zeros_like(image_gray)
    region_of_interest= cv2.fillPoly(blank, ROI,255)
    region_of_interest_image= cv2.bitwise_and(image_gray, region_of_interest)    
    x1,x2,y1,y2 = (0,0,0,0)
    for linha in range(0,len(region_of_interest_image)):
        for coluna in range(0,len(region_of_interest_image[0])):
            if region_of_interest_image[linha][coluna] != 0:
                          
                x1 = linha
                y1 = coluna
                break
        if x1 != 0:
            break
    
    for linha in range(len(region_of_interest_image)-1,0,-1):
        for coluna in range(len(region_of_interest_image[0])-1,0,-1):
            if region_of_interest_image[linha][coluna] != 0:
                          
                x2 = linha
                y2 = coluna
                break
        if x2 != 0:
            break

    slicedImage = image[x1:x2,y1:y2]
    print(slicedImage)
    fs = 30
    low_freq = 0.4
    high_freq = 4
    nyq = 0.5*fs
    low = low_freq/nyq
    high = high_freq/nyq 
    num,den = scipy.signal.butter(2,[0.4,4],'bandpass',30)  
    filtered = scipy.signal.filtfilt(num,den,slicedImage)




#Mostrando o resultado

cv2.imshow('image', image)
cv2.imshow('Region of Interest', slicedImage)
cv2.imshow('Filtragem',filtered)
cv2.waitKey()
cv2.destroyAllWindows()






#teste 2

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)



# Desenhando retangulos ao redors de rostos
for (x, y, w, h) in faces:
    cv2.rectangle(image2, (x, y), (x + w, y + h),(0,255,0), 2)
    faceROI = image2[y:y+h,x:x+w]
    eyes = eyeCascade.detectMultiScale(faceROI)

    l1 = x + eyes[0][0]+eyes[0][2]//2
    l2 = x + eyes[1][0]+eyes[1][2]//2
    c1 = y 
    c2 = y + eyes[1][1]+eyes[1][3]//2-int(round((eyes[0][2] + eyes[0][3]) * 0.45))
    slicedImage = image[c1:c2,l1:l2]

    for (x2, y2, w2, h2) in eyes:
        eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
        
        radius = int(round((w2 + h2) * 0.25))
        image2 = cv2.rectangle(image2,(eye_center[0]-radius,eye_center[1]-radius), (eye_center[0]+radius,eye_center[1]+radius), (255, 0, 0), 2)
       
    


#Mostrando o resultado

#cv2.imshow("Faces found", image)
#cv2.imshow("Faces found2", image2)
#cv2.imshow("roi", slicedImage)
#cv2.waitKey(0)

#testando com pessoas

'''
pessoa1 = cv2.imread("rosto_1.jpg",cv2.IMREAD_COLOR)
hsv1 = cv2.cvtColor(pessoa1,cv2.COLOR_BGR2HSV)
pessoa2 = cv2.imread("rosto_3.jpg",cv2.IMREAD_COLOR)
hsv2 = cv2.cvtColor(pessoa2,cv2.COLOR_BGR2HSV)
pessoa3 = cv2.imread("rosto_4.png",cv2.IMREAD_COLOR)
hsv3 = cv2.cvtColor(pessoa3,cv2.COLOR_BGR2HSV)
'''

#Pessoa 1
'''
mask = cv2.inRange(hsv1,lower_range,upper_range)
cv2.imshow('image_window_name', pessoa1)
cv2.imshow('mask_window_name', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#Pessoa 2
'''
mask = cv2.inRange(hsv2,lower_range,upper_range)
cv2.imshow('image_window_name', pessoa2)
cv2.imshow('mask_window_name', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#Pessoa 3
'''
mask = cv2.inRange(hsv3,lower_range,upper_range)
cv2.imshow('image_window_name', pessoa3)
cv2.imshow('mask_window_name', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''




#plt.figure("pyplot galáxia")
#img = cv2.imread("galaxia.jpg")
#plt.imshow(img)
#plt.show()


#treinamento com video 

vid = cv2.VideoCapture("modelo1treino.mp4")
#vid = cv2.VideoCapture(0)

while(1):
  _, image = vid.read()
  _, image2 = vid.read()
  _, image3 = vid.read()

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60),
    flags = cv2.CASCADE_SCALE_IMAGE
    )
  for (x, y, w, h) in faces:
        cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 255, 0), 2)

  for (x, y, w, h) in faces:
        
        #cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ROI= np.array([[(x+0.25*w,y),(x+0.8*w,y),(x+0.8*w,y+0.2*h),(x+0.25*w,y+0.2*h)]], dtype= np.int32)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blank= np.zeros_like(image_gray)
        region_of_interest= cv2.fillPoly(blank, ROI,255)
        region_of_interest_image= cv2.bitwise_and(image_gray, region_of_interest)    
  for (x, y, w, h) in faces:
    cv2.rectangle(image3, (x, y), (x + w, y + h),(0,255,0), 2)
    faceROI = image3[y:y+h,x:x+w]
    eyes = eyeCascade.detectMultiScale(faceROI)
#    if len(eyes) > 1:
#        l1 = x + eyes[0][0]+eyes[0][2]//2
#        l2 = x + eyes[1][0]+eyes[1][2]//2
#        c1 = y 
#        c2 = y + eyes[1][1]+eyes[1][3]//2-int(round((eyes[0][2] + eyes[0][3]) * 0.5))
#        print(l1,l2,c1,c2)
#    else:
    l1 = x+round(0.3*w)
    l2 = x+round(0.75*w)
    c1 = y
    c2 = y + round(0.2*h)
    slicedImage3 = image[c1:c2,l1:l2]
    
  #cv2.imshow("Faces found", image2)
  #cv2.imshow("Região de interesse",slicedImage3)
  #cv2.imshow("ROI",region_of_interest_image)
  #cv2.waitKey(5)
#cv2.destroyAllWindows()

#  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#  lower_range = np.array([110,50,50])
#  upper_range = np.array([130,255,255])
#  mask = cv2.inRange(hsv,lower_range,upper_range)
#  cv2.imshow('image_window_name',image)
#  cv2.imshow('mask_window_namw',mask)
#  cv2.waitKey(5) 
#cv2.destroyAllWindows()


#vid = cv2.VideoCapture("Data_Inidia_video10_forehead(1)")

#while(vid.isOpened()):
#  _, image = vid.read()
#  image = cv2.resize(image,(700,700))
#  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#  mask = cv2.inRange(hsv,green_lower_range,green_upper_range)
#  cv2.imshow('image_window_name',image)
#  cv2.imshow('mask_window_namw',mask)
#  cv2.waitKey(5) 

#vid.release()
#cv2.destroyAllWindows()

