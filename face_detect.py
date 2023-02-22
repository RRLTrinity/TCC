import cv2 
import os
import dlib

def ajuste_de_bordas(image,rectangle):

    #Definindo os vértices dos retangulos ao redor do rosto
    x_i = rectangle.left()
    x_f = rectangle.right()
    y_i = rectangle.top()
    y_f = rectangle.bottom()

    #Garantindo que as bordas estejam dentro dos limites da imagem
    x_i = max(0,x_i)
    x_f = min(x_f,image.shape[1])
    y_i = max(0,y_i)
    y_f = min(y_f,image.shape[0])

    return (x_i,x_f,y_i,y_f)

def conv_param_dlib_cv(image, rectangle):

    #ajustando bordas
    (x_i,x_f,y_i,y_f) = ajuste_de_bordas(image,rectangle)

    #convertendo os parâmetros

    w = x_f-x_i
    h = y_f-y_i

    return (x_i,y_i,w,h)

def facedetection(image):
    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    cascPatheyes = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    eyeCascade = cv2.CascadeClassifier(cascPatheyes)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

def roi(image):
    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    cascPatheyes = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    eyeCascade = cv2.CascadeClassifier(cascPatheyes)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        l1 = x+round(0.3*w)
        l2 = x+round(0.75*w)
        c1 = y
        c2 = y + round(0.2*h)
        slicedImage = image[c1:c2,l1:l2]
    return slicedImage

