import matplotlib.pyplot as plt
import shapely.geometry
import cv2
import numpy as np
import util_contours
import os
import traceback
import facial_landmarks

#import topojson as tp
#import geopandas as gpd

#BGR!!!!
#hair: 5,56,182
#skin: 121,141,205 #darker = 106,125,171 #contours: 118, 113, 168
	#NO right arm: 118, 135, 168
#left leg:72,72,72 #right leg:2,2,2
#shirt:49,32,46 #shoulder:44,34,42 
#ball:89,69,4
#pupils: 61, 71, 118
color_assignment = { #color from original segmentation (grayscale) to final color
    170: (121,141,205,255), #right arm
    174: (34,34,34,255),#left leg
    162: (89,69,4,255),#ball
    179: (121,141,205,255),#face 
    196: (22,22,22,255),#right leg
    207: (49,32,46,255),#shirt
    255: (5,56,182,255),#hair
    239: (44,34,42,255)#right shoulder
}

'''
color_assignment = { #color from original segmentation (grayscale) to final color
    162: (0,64,158,255)
}
'''

def simplify(opencvContour, tolerance = 4.0):#5.0 , preserve_topology=False
    """ Simplify a polygon with shapely.
    Polygon: ndarray
        ndarray of the polygon positions of N points with the shape (N,2)
    """
    polygon = np.squeeze(opencvContour)
    poly = shapely.geometry.Polygon(polygon)
    poly_s = poly.simplify(tolerance=tolerance, preserve_topology=False)
    # convert it back to numpy
    coords = np.array(poly_s.boundary.coords[:])
    #Convert shapely polygon (N, 2) to opencv contour (N-1, 1, 2)
    opencvContourSimplified = coords.reshape((-1,1,2)).astype(np.int32)    
    return opencvContourSimplified

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

def getContours(im):
    height, width = im.shape[:2]
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    contours_raw = []
    contours_simplified = []
    colors = []

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
            
            '''
            #remove anything outside the contour
            mask = cropContours(mask, contours)

            cv2.imshow("title", mask)
            cv2.waitKey() 

            #find contours again
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            #dilate 1 pixel (to avoid gaps between simplified contours)
            kernel = np.ones((4, 4), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            #find contours again
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            '''

            #Retrieval modes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
            #RETR_TREE, RETR_LIST, RETR_EXTERNAL
            #Contour approx mode: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
            #CHAIN_APPROX_NONE, 
            #offset: 
            for j, contour in enumerate(contours):
                if cv2.contourArea(contour) > 10:
                    print("Color number "+str(colorNum)+"="+str(color))
                    #dilate 1 pixel (to avoid gaps between simplified contours)
                    #remove anything outside the contour
                    part_mask = cropContours(mask, contour)
                    kernel = np.ones((4, 4), np.uint8)
                    part_mask = cv2.dilate(part_mask, kernel, iterations=1)
                    #cv2.imshow("title", part_mask)
                    #cv2.waitKey() 

                    #find contours again
                    ret, thresh = cv2.threshold(part_mask, 127, 255, 0)
                    image, contours_dilated, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    max_contour = max(contours_dilated, key = cv2.contourArea)

                    contours_raw.append(max_contour)
                    colors.append(color)
                    test = np.zeros_like(imgray)
                    #cv2.drawContours(test, [max_contour], contourIdx=0, color=(100,200,100), thickness=2)
                    #cv2.imshow("title", test)
                    #cv2.waitKey() 
                    totalContours = totalContours+1                          
            colorNum = colorNum+1
    print("Found "+str(colorNum)+" colors")
    print("Found "+str(totalContours)+" contours")
    return contours_raw, colors

def cropContours(im, contour):
    im_res = np.zeros_like(im)
    cv2.fillPoly(im_res, pts =[contour], color=(255,255,255))
    return im_res

def simplifyContours(contours):
    contours_simplified = []
    for contour in contours:
        try:
            simplifiedContour = simplify(contour)
            contours_simplified.append(simplifiedContour)
        except:
            print("Contour discarded as contains multi-part geometries")
            print(traceback.format_exc())
            print("Using original contour without simplification")
            contours_simplified.append(contour)
    return contours_simplified

#not used
def shadow_contour(contour):
	contour_shifted = np.copy(contour)
	for i in range(len(contour_shifted)):
		contour_shifted[i][0] = contour_shifted[i][0] + 5
	return contour_shifted
#not used
def darkColor(color):
	return (abs(color[0]/2), abs(color[1]/2), abs(color[2]/2), 255)

def fillContours(contours, colors, imcolor):
	for i, contour in enumerate(contours):
		#shadow = shadow_contour(contour)
		#cv2.fillPoly(imcolor, pts =[shadow], color=darkColor(color_assignment[colors[i]]))
		cv2.fillPoly(imcolor, pts =[contour], color=color_assignment[colors[i]])

	imcolor_pixelated = pixelate(imcolor, 512, 512)
	return imcolor_pixelated

def change_brightness(img, value=-30):
    _, _, _, a_channel = cv2.split(img)
    #img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    img = np.dstack((img, a_channel))
    return img



def addShadow(imcolor):    
    imcolor_result_shadow = imcolor.copy()
    height, width = imcolor_result_shadow.shape[:2]
    offsetx = 10
    offsety = 0
    M = np.float32([[1, 0, offsetx], [0, 1, offsety]])
    dst_mat = np.zeros((height, width, 4), np.uint8)
    size = (width, height)
    '''
    dst_mat = np.zeros((height, width, 4), np.uint8)
	# 回転させたい角度（正の値は反時計回り）
    size = (width, height)
    center = (width/2,height/2)
    angle = -45.0
	# 拡大比率
    scale = 0.5
	# 回転変換行列の算出
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    imcolor_result_shadow = cv2.warpAffine(imcolor_result_shadow, rotation_matrix, size, dst_mat, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    '''
    #imcolor_result_shadow = cv2.warpAffine(imcolor_result_shadow, M, size, dst_mat, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    imcolor_result_shadow = cv2.warpAffine(imcolor_result_shadow, M, size, dst_mat)
    #alpha = 1.5 # Contrast control
    #beta = 10 # Brightness control
    #imcolor_result_shadow = cv2.convertScaleAbs(imcolor_result_shadow, alpha=alpha, beta=beta)
    imcolor_result_shadow = change_brightness(imcolor_result_shadow)
    #imcolor_result_shadow = imcolor + imcolor_result_shadow

    #imcolor_result_shadow = cv2.addWeighted(imcolor, 1, imcolor_result_shadow, 1, 0) 
    #imcolor[imcolor[:, :, 1:].all(axis=-1)] = 0
	#img2[img2[:, :, 1:].all(axis=-1)] = 0



    '''
    imcolor = np.zeros_like(imcolor)
    #imcolor = addAlpha(imcolor)
    cv2.circle(imcolor, (100,100), 100, (255,0,0,255), thickness=-1)

    imcolor_result_shadow = np.zeros_like(imcolor_result_shadow)
    #imcolor_result_shadow = addAlpha(imcolor_result_shadow)
    cv2.circle(imcolor_result_shadow, (140,140), 100, (0,255,0,255), thickness=-1)
    '''
    '''
    gray2 = cv2.cvtColor(imcolor_result_shadow, cv2.COLOR_BGR2GRAY)
    thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    result1 = cv2.bitwise_and(imcolor_result_shadow, imcolor_result_shadow, mask=thresh2)
    result2 = cv2.bitwise_and(imcolor, imcolor, mask=255-thresh2)
    imcolor_result_shadow = cv2.add(result1, result2)
    '''
    imcolor_result_shadow = overlay(imcolor, imcolor_result_shadow)
    
    #imcolor_result_shadow = result1 + imcolor_result_shadow

    #imcolor_result_shadow = cv2.addWeighted(imcolor, 1, imcolor_result_shadow, 1, 0) 
    
    return imcolor_result_shadow
def overlay(bottomImage, topImage):
    gray2 = cv2.cvtColor(bottomImage, cv2.COLOR_BGR2GRAY)
    thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    result1 = cv2.bitwise_and(bottomImage, bottomImage, mask=thresh2)
    result2 = cv2.bitwise_and(topImage, topImage, mask=255-thresh2)
    return cv2.add(result1, result2)
   
def drawContours(contours, colors, imcolor):
    for i, contour in enumerate(contours):
        cv2.drawContours(imcolor, [contour], contourIdx=0, color=color_assignment[colors[i]], thickness=1)        
    #imcolor_pixelated = pixelate(imcolor, 512, 512)
    return imcolor

'''
        MAIN
'''
inputpathOriginal = '/Users/rtous/DockerVolume/seg4art/data/scenes/ruben2/imagesFull'
inputpath = '/Users/rtous/DockerVolume/seg4art/data/scenes/ruben2/samtrack'
outputpath = '/Users/rtous/DockerVolume/seg4art/data/scenes/ruben2/out_opencv/'
outputpath_contours = '/Users/rtous/DockerVolume/seg4art/data/scenes/ruben2/out_opencv_contours/'

#inputpath = '/Users/rtous/DockerVolume/seg4art/data/scenes/tiktok2/out_pngs'
#outputpath = '/Users/rtous/DockerVolume/seg4art/data/scenes/tiktok2/out_opencv/'
#outputpath_contours = '/Users/rtous/DockerVolume/seg4art/data/scenes/tiktok2/out_opencv_contours/'

if not os.path.exists(outputpath):
   os.makedirs(outputpath)
if not os.path.exists(outputpath_contours):
   os.makedirs(outputpath_contours)
for filename in sorted(os.listdir(inputpath)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
        print("process("+inputpath+"/"+filename+", "+outputpath+")")
        #Read image with opencv
        im = cv2.imread(os.path.join(inputpath, filename))
        assert im is not None, "file could not be read, check with os.path.exists()"
       
        #add a border (to avoid edge contours to be discarded)
        #im = cv2.copyMakeBorder(im, 50, 50, 50, 50, cv2.BORDER_CONSTANT, None, value = 0) 

        #cv2.imshow("title", im)
        #cv2.waitKey()
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        #find relevant contours
        contours_raw, colors = getContours(im)

        #align close contours
        #contours_raw = util_contours.fillGaps(contours_raw)

        #simplify
        contours_simplified = simplifyContours(contours_raw)

        #draw contours, pixelate and write file
        imcolor = np.zeros_like(im)
        imcolor = addAlpha(imcolor)
        imcolor_result = fillContours(contours_simplified, colors, imcolor)

        #add shadows
        imcolor_result = addShadow(imcolor_result)

        #draw face elements
        facial_landmarks.faceFromPath(os.path.join(inputpathOriginal, filename), imcolor_result)
        
        #write image
        cv2.imwrite(os.path.join(outputpath, filename), imcolor_result)
        print("cv2.imwrite("+os.path.join(outputpath, filename)+")")

        imcolor_contours = np.zeros_like(im)
        imcolor_contours = addAlpha(imcolor_contours)
        imcolor_contours_result = drawContours(contours_raw, colors, imcolor_contours)    
        cv2.imwrite(os.path.join(outputpath_contours, filename), imcolor_contours_result)
