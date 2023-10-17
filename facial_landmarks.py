import matplotlib.pyplot as plt
import shapely.geometry
import cv2
import numpy as np
import util_contours
import os
from imutils import face_utils
import imutils
import dlib

#From https://datagen.tech/guides/face-recognition/facial-landmarks/

def process(inputpath, outputpath):
    print("process("+inputpath+", "+outputpath+")")
    #Read image with opencv
    im = cv2.imread(inputpath)
    assert im is not None, "file could not be read, check with os.path.exists()"
    height, width = im.shape[:2]

    
    # initialize built-in face detector in dlib
    detector = dlib.get_frontal_face_detector()
    # initialize face landmark predictor
    PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat"#https://github.com/davisking/dlib-models
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    #resize to width 500
    #im_resized = cv2.resize(im, (500, 500), interpolation= cv2.INTER_LINEAR)
    image = imutils.resize(im, width=500)
    #convert it to grayscale
    im_resized_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(im_resized_gray, 1)
    #for each face
    for (i, rect) in enumerate(rects):
        # predict facial landmarks in image and convert to NumPy array
        shape = predictor(im_resized_gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convert to OpenCV-style bounding box
        #for (x, y) in shape:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number and draw facial landmarks on the image
        #cv2.putText(image, “Face #{}”.format(i + 1), (x – 10, y – 10),
        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        # show the resulting output image

        cv2.imshow("Output", image)
        cv2.waitKey(0)

# load input image, resize it, and convert it to grayscale



    #cv2.imshow("title", im)
    #cv2.waitKey()
    #height, width = im.shape[:2]
    #imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #imcolor = np.zeros_like(im)
    #imcolor = addAlpha(imcolor)

    
    #print("cv2.imwrite("+outputpath+")")
    #print("Found "+str(colorNum)+" colors")
    #print("Found "+str(totalContours)+" contours")
    #cv2.imwrite(outputpath, imcolor)

inputpath = '/Users/rtous/DockerVolume/seg4art/data/scenes/ruben2/imagesFull'
outputpath = '/Users/rtous/DockerVolume/seg4art/data/scenes/ruben2/out_face/'
for filename in sorted(os.listdir(inputpath)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        process(os.path.join(inputpath, filename), os.path.join(outputpath, filename))

