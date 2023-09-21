import matplotlib.pyplot as plt
import shapely.geometry
import cv2
import numpy as np
import util_contours
#import topojson as tp
#import geopandas as gpd

colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,255)]  

def simplify(polygon, tolerance = 5.0):
    """ Simplify a polygon with shapely.
    Polygon: ndarray
        ndarray of the polygon positions of N points with the shape (N,2)
    """
    poly = shapely.geometry.Polygon(polygon)
    poly_s = poly.simplify(tolerance=tolerance, preserve_topology=False)
    # convert it back to numpy
    return np.array(poly_s.boundary.coords[:])

#Read image with opencv
im = cv2.imread('/Users/rtous/DockerVolume/seg4art/data/scenes/tiktok2/out_pngs/00000.png')
assert im is not None, "file could not be read, check with os.path.exists()"
cv2.imshow("title", im)
cv2.waitKey()
height, width = im.shape[:2]
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imcolor = np.zeros_like(im)

'''
SEEMS UNNECESSARY
#remove anything except the biggest contour
mask = np.zeros_like(imgray) # Create mask where white is what we want, black otherwise
#find outer contour
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask, contours, 0, 255, -1) # Draw filled contour in mask
out = np.zeros_like(im) # Extract out the object and place into output image
out[mask == 255] = im[mask == 255]
cv2.imshow("title", im)
cv2.waitKey()
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
'''

#polygons = []


#split image in C color regions (with a minimum of 1000 pixels)
selected_contours = []
contours_simplified = [] 
colorNum = 0
unique = np.unique(imgray)
for i, color in enumerate(unique):
    mask = np.zeros_like(imgray)
    mask[imgray == color] = 255

    #cv2.imshow("title", mask)
    #cv2.waitKey()
    #ret, thresh = cv2.threshold(mask, 127, 255, 0)
    #image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #area = cv2.contourArea(contours[0])
    area = cv2.countNonZero(mask)
    if area > 1000 and area < height*width/2: #avoid the frame contour
        #print("Found big area")
        #cv2.imshow("title", mask)
        #cv2.waitKey()
        #split color mask in N contours (with a minimum of area > 10)
        #selected_contours.append(contours[0])
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        #image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #Retrieval modes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
        #RETR_TREE, RETR_LIST, RETR_EXTERNAL
        #Contour approx mode: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
        #CHAIN_APPROX_NONE, 
        #offset: 
        for j, contour in enumerate(contours):
            if cv2.contourArea(contour) > 10:
                print("Mask: "+str(i)+"/ Contour "+str(j))
                selected_contours.append(contour)
                #contourImg = np.zeros_like(imgray)
                mask = np.zeros_like(imgray)
                cv2.drawContours(mask, [contour], contourIdx=0, color=(100,200,100), thickness=3)
                cv2.imshow("title", mask)
                cv2.waitKey()
                #poly = shapely.geometry.Polygon(np.squeeze(contour))
                #polygons.append(poly)
                #simplifiedContour = simplify(np.squeeze(contours[0]))
                try:
                    simplifiedContour = simplify(np.squeeze(contour))
                    #Convert shapely polygon (N, 2) to opencv contour (N-1, 1, 2)
                    simplifiedContourReshaped = np.array(simplifiedContour).reshape((-1,1,2)).astype(np.int32)
                    cv2.fillPoly(mask, pts =[simplifiedContourReshaped], color=(255,255,255))
                    cv2.fillPoly(imcolor, pts =[simplifiedContourReshaped], color=colors[colorNum])
                    contours_simplified.append(simplifiedContourReshaped)
                    cv2.imshow("title", mask)
                    cv2.waitKey()
                    
                except:
                    print("Contour discarded as contains multi-part geometries")
        colorNum = colorNum+1
cv2.imshow("title", imcolor)
cv2.waitKey()
'''
#Some contours have similar boundaries
#Here I try to unify them 
#Currently disabled

#debug_contours = []
#debug_contours.append(selected_contours[2])
#debug_contours.append(selected_contours[9])


#mask = np.zeros_like(imgray)
#cv2.drawContours(mask, [selected_contours[0]], contourIdx=0, color=(255,200,100), thickness=1)
#cv2.imshow("title", mask)
#cv2.waitKey()
#cv2.drawContours(mask, [selected_contours[1]], contourIdx=0, color=(100,200,100), thickness=1)
#cv2.imshow("title", mask)
#cv2.waitKey()



print("purging!")
purged_contours = util_contours.extractOverlappingContours(selected_contours)
print("purging again!")
purged_contours = util_contours.extractOverlappingContours(purged_contours)


print("Found "+str(len(purged_contours))+" intersecting contours:")
#print("Contours=", purged_contours)


for i, contour in enumerate(purged_contours):
    print("drawing contour num ", i)
    mask = np.zeros_like(imgray)
    cv2.drawContours(mask, [contour], contourIdx=0, color=(100,255,100), thickness=1)
    cv2.imshow("title", mask)
    cv2.waitKey()    
'''

'''
for i, contour in enumerate(contours_simplified):
    print("drawing contour num ", i)
    mask = np.zeros_like(imgray)

    cv2.drawContours(mask, contour, contourIdx=0, color=(255,200,100), thickness=1)

    cv2.fillPoly(mask, pts =[contour], color=(255,255,255))
    #cv2.drawContours(mask, [contour], contourIdx=0, color=(100,255,100), thickness=2)
    cv2.imshow("title", mask)
    cv2.waitKey() 
'''   

