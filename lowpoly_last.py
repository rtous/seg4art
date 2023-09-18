import matplotlib.pyplot as plt
import shapely.geometry
import cv2
import numpy as np
import util_contours
#import topojson as tp
#import geopandas as gpd 

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
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

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
    if area > 1000:
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
                #cv2.drawContours(mask, [contour], contourIdx=0, color=(100,200,100), thickness=3)
                #cv2.imshow("title", mask)
                #cv2.waitKey()
                #poly = shapely.geometry.Polygon(np.squeeze(contour))
                #polygons.append(poly)

debug_contours = []
#debug_contours.append(selected_contours[2][10:50])

#debug_contours.append(selected_contours[9][75:100])
#debug_contours.append(selected_contours[2][90:115])

#debug_contours.append(selected_contours[9][250:300])
#debug_contours.append(selected_contours[2][190:200])


debug_contours.append(selected_contours[2])
debug_contours.append(selected_contours[9])


mask = np.zeros_like(imgray)
cv2.drawContours(mask, [debug_contours[0]], contourIdx=0, color=(255,200,100), thickness=1)
cv2.imshow("title", mask)
cv2.waitKey()
#mask = np.zeros_like(imgray)
cv2.drawContours(mask, [debug_contours[1]], contourIdx=0, color=(100,200,100), thickness=1)
cv2.imshow("title", mask)
cv2.waitKey()



purged_contours = util_contours.extractOverlappingContours(debug_contours)
print("Found "+str(len(purged_contours))+" intersecting contours:")
print("Contours=", purged_contours)


for i, contour in enumerate(purged_contours):
    print("drawing contour num ", i)
    mask = np.zeros_like(imgray)
    cv2.drawContours(mask, contour, contourIdx=0, color=(100,255,100), thickness=1)
    cv2.imshow("title", mask)
    cv2.waitKey()    

'''                
multipoly = shapely.geometry.MultiPolygon(polygons=polygons)
multipoly_simplified = multipoly.simplify(tolerance=5, preserve_topology=True)

polygon_array = np.array(multipoly_simplified.boundary.coords[:])
contour_opencv= np.array(polygon_array).reshape((-1,1,2)).astype(np.int32)
cv2.drawContours(im, [contour_opencv], 0, (0,255,0), 3)
cv2.imshow("title", im)
cv2.waitKey()
'''


