import matplotlib.pyplot as plt
import shapely.geometry
import cv2
import numpy as np
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
        print("Found big area")
        cv2.imshow("title", mask)
        cv2.waitKey()
        #split color mask in N contours (with a minimum of area > 10)
        #selected_contours.append(contours[0])
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 10:
                print("Found big countour")
                #contourImg = np.zeros_like(imgray)
                cv2.drawContours(mask, [contour], contourIdx=0, color=(100,200,100), thickness=3)
                cv2.imshow("title", mask)
                cv2.waitKey()


#Get opencv contours
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#WARN: This version returns 3 values
#https://docs.opencv.org/3.4.13/d4/d73/tutorial_py_contours_begin.html


#Draw all contours with opencv (changing the -1 can draw just one)
#cv2.drawContours(im, contours, -1, (0,255,0), 3)

#Pick one contour
c0 = contours[0]
#print("opencv contour shape:", c0.shape)
#print("opencv contour:")
#print(c0)
'''
#Draw the selected contour
cv2.drawContours(im, [c0], 0, (0,255,0), 3)
cv2.imshow("title", im)
cv2.waitKey()
'''

# Simplify the contour with shapely
simplifiedContour = simplify(np.squeeze(c0)) 
#print("shapely contour shape:", simplifiedContour.shape)
#print("shapely contour:")
#print(simplifiedContour)

#simplify all:
#contours_s = []
#for c in contours:
#	contours_s.append(simplify(np.squeeze(c)))

'''
#Convert shapely polygon (N, 2) to opencv contour (N-1, 1, 2)
simplifiedContourReshaped = np.array(simplifiedContour).reshape((-1,1,2)).astype(np.int32)

#Draw with opencv
cv2.drawContours(im, [simplifiedContourReshaped], 0, (0,255,0), 3)
cv2.imshow("title", im)
cv2.waitKey()
'''

#All contours

contours_s = []
for c in contours:
    contours_s.append(simplify(np.squeeze(c)))

#With a shapely multipoly
polygons = []
for c in contours:
    print("added contour to polygons")
    poly = shapely.geometry.Polygon(np.squeeze(c))
    polygons.append(poly)
multipoly = shapely.geometry.MultiPolygon(polygons=polygons)
multipoly_simplified = multipoly.simplify(tolerance=5, preserve_topology=True)

polygon_array = np.array(multipoly_simplified.boundary.coords[:])
contour_opencv= np.array(polygon_array).reshape((-1,1,2)).astype(np.int32)
cv2.drawContours(im, [contour_opencv], 0, (0,255,0), 3)
cv2.imshow("title", im)
cv2.waitKey()

#Try to visualize with opencv
'''
for polygon in multipoly_simplified:  # same for multipolygon.geoms
    polygon_array = np.array(poly_s.boundary.coords[:])
    contour_opencv= np.array(polygon_array).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(im, [contour_opencv], 0, (0,255,0), 3)
cv2.imshow("title", im)
cv2.waitKey()
'''

#poly = shapely.geometry.MultiPolygon(contours_s)

#MultiPolygon(polygons=None)


'''
#NO FUNCIONA JA QUE NO PUC INSTALAR GEOPANDAS
gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[simplifiedContour])
print(gdf)
topo = tp.Topology(gdf.to_crs({'init':'epsg:3857'}), prequantize=False)
simple = topo.toposimplify(1).to_gdf()
# optional
simple.plot()
'''


