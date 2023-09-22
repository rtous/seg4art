import matplotlib.pyplot as plt
import shapely.geometry
import cv2
import numpy as np
import util_contours
import os
import traceback

#import topojson as tp
#import geopandas as gpd

#BGR!!!!

color_assignment = { #color from original segmentation (grayscale) to final color
    170: (0,64,158,255), 
    174: (64,49,47,255),
    239: (111,128,191,255)
}

def simplify(polygon, tolerance = 4.0):#5.0 , preserve_topology=False
    """ Simplify a polygon with shapely.
    Polygon: ndarray
        ndarray of the polygon positions of N points with the shape (N,2)
    """
    poly = shapely.geometry.Polygon(polygon)
    poly_s = poly.simplify(tolerance=tolerance, preserve_topology=False)
    # convert it back to numpy
    return np.array(poly_s.boundary.coords[:])

def addAlpha(img):
    #b_channel, g_channel, r_channel = cv2.split(img)
    #alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
    #img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    #img_BGRA = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    b_channel, g_channel, r_channel = cv2.split(img)
    #alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 0
    alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def pixelate(input, w, h): # w,h  Desired "pixelated" size
    height, width = input.shape[:2]
    # Resize input to "pixelated" size
    temp = cv2.resize(input, (w, h),  interpolation=cv2.INTER_NEAREST)#cv2.INTER_LINEAR antialiasing
    # Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output

def process(inputpath, outputpath):
    print("process("+inputpath+", "+outputpath+")")
    #Read image with opencv
    im = cv2.imread(inputpath)
    assert im is not None, "file could not be read, check with os.path.exists()"
    #cv2.imshow("title", im)
    #cv2.waitKey()
    height, width = im.shape[:2]
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imcolor = np.zeros_like(im)
    imcolor = addAlpha(imcolor)

    #split image in C color regions (with a minimum of 1000 pixels)
    selected_contours = []
    contours_simplified = [] 
    colorNum = 0
    totalContours = 0
    unique = np.unique(imgray)
    for i, color in enumerate(unique):
        mask = np.zeros_like(imgray)
        mask[imgray == color] = 255
        area = cv2.countNonZero(mask)
        if area > 1000 and area < height*width/2: #avoid the frame contour
            #split color mask in N contours (with a minimum of area > 10)
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #Retrieval modes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
            #RETR_TREE, RETR_LIST, RETR_EXTERNAL
            #Contour approx mode: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
            #CHAIN_APPROX_NONE, 
            #offset: 
            for j, contour in enumerate(contours):
                if cv2.contourArea(contour) > 10:
                    print("Color number "+str(colorNum)+"="+str(color))
                    #mask = np.zeros_like(imgray)
                    #cv2.drawContours(mask, [contour], contourIdx=0, color=(100,200,100), thickness=2)
                    #cv2.imshow("title", mask)
                    #cv2.waitKey()
                    #use a try here because simplify may fail in some geometries
                    try:
                        simplifiedContour = simplify(np.squeeze(contour))
                        #Convert shapely polygon (N, 2) to opencv contour (N-1, 1, 2)
                        simplifiedContourReshaped = np.array(simplifiedContour).reshape((-1,1,2)).astype(np.int32)
                        #cv2.fillPoly(mask, pts =[simplifiedContourReshaped], color=(255,255,255))
                        cv2.fillPoly(imcolor, pts =[simplifiedContourReshaped], color=color_assignment[color])
                        #cv2.imshow("title", mask)
                        #cv2.waitKey()
                        totalContours = totalContours+1
                    except:
                        print("Contour discarded as contains multi-part geometries")
                        print(traceback.format_exc())
                        print("Using original contour without simplification")
                        cv2.fillPoly(imcolor, pts =[contour], color=color_assignment[color])
                        
            colorNum = colorNum+1
    imcolor = pixelate(imcolor, 512, 512)
    #cv2.imshow("title", imcolor)
    #cv2.waitKey()
    print("cv2.imwrite("+outputpath+")")
    print("Found "+str(colorNum)+" colors")
    print("Found "+str(totalContours)+" contours")
    cv2.imwrite(outputpath, imcolor)

inputpath = '/Users/rtous/DockerVolume/seg4art/data/scenes/tiktok2/out_pngs'
outputpath = '/Users/rtous/DockerVolume/seg4art/data/scenes/tiktok2/out_opencv/'
for filename in sorted(os.listdir(inputpath)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
        process(os.path.join(inputpath, filename), os.path.join(outputpath, filename))

