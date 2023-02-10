import cv2 
import os
import dlib

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

