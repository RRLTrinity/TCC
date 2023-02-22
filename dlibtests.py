import cv2
import dlib 
import argparse
import time
import imutils

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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str,
	default="mmod_human_face_detector.dat",
	help="path to dlib's CNN face detector model")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())

#define the model face_detector()
detector1 = dlib.get_frontal_face_detector()
detector2 = dlib.cnn_face_detection.py(mmod_human_face_detector.dat)

#Select the image, resize it and then convert to rgb

image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
image1 = imutils.resize(image, width=600)
image2 = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#dlib(HOG+linearsvm)
print("[Aviso] realizando a detecção facial do dlib pelo método HOG + linear SVM...")
começo = time.time()
rects = detector1(rgb, args["upsample"])
fim = time.time()
print("[Aviso]] Tempo de excução da detecção facial {:.4f} segundos".format(fim - começo))

#adjusting the boxes
boxes = [conv_param_dlib_cv(image, r) for r in rects]

#extracting the parameters
for (x, y, w, h) in boxes:
	# drawing the boxes in the image
	cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 145), 2)

#dlib(MMOD CNN)
print("[Aviso] realizando a detecção facial do dlib pelo método MMOD CNN...")
começo = time.time()
rects = detector2(rgb, args["upsample"])
fim = time.time()
print("[Aviso]] Tempo de excução da detecção facial {:.4f} segundos".format(fim - começo))

#adjusting the boxes
boxes = [conv_param_dlib_cv(image, r.rect) for r in rects]

#extracting the parameters
for (x, y, w, h) in boxes:
	# drawing the boxes in the image
	cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 145), 2)
         
# show the outputs images
cv2.imshow("Output1", image1)
cv2.imshow("Output2", image2)
cv2.waitKey(0)