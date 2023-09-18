import matplotlib.pyplot as plt
import shapely.geometry
import cv2
import numpy as np

def extractOverlappingContours(contours):
    '''
    When coincident fragments, split the contours
    '''
    resulting_contours = []
    '''
    contours = [
        np.array([[[7,7]],[[1,1]],[[2,2]],[[3,3]],[[7,7]],[[7,7]],[[8,8]],[[9,9]],[[7,7]]], dtype=np.int32), 
        np.array([[[5,5]],[[1,1]],[[2,2]],[[3,3]],[[5,5]],[[8,8]],[[9,9]],[[5,5]]], dtype=np.int32),
    ]
    '''
    
    print("extractOverlappingContours over len(contours)= ",len(contours))
    for i in range(len(contours)):
        for j in range(i+1, len(contours)):
            print("comparing contour num "+str(i)+" with num"+str(j))
            print("*********************")
            c1 = contours[i]
            c2 = contours[j]
            '''
            if len(c1) >= len(c2):
                rc  = substractContours(c1, c2)
            else:
                rc  = substractContours(c2, c1)
            '''
            rc  = substractContours(c2, c1)
            for c in rc:
                resulting_contours.append(rc)
    return resulting_contours             

def substractContours(c1, c2):
    '''
    - Order does not matter
    - There can be more than one matching segments
    - Assumens there're only occurrence of each segment in on contour
    '''
    #print("c1=", c1)
    #print("c2=", c2)
    current_contour = np.array(None)
    new_contour = None
    final_contours = []
    idxc1 = 0
    idxc2 = 0
    overlapping = False
    while idxc1 < len(c1):
        #print("iter: idxc1="+str(idxc1)+", idxc2="+str(idxc2))
        print("\tc1[idxc1]="+str(c1[idxc1])+", c2[idxc2]="+str(c2[idxc2]))
        #print("check idxc1="+str(idxc1)+" vs "+str(idxc2))
        #if np.array_equal(c1[idxc1], c2[idxc2]) and not overlapping:
        #if np.allclose(c1[idxc1], c2[idxc2], rtol=0.1) and not overlapping: 
        if similarPoints(c1[idxc1], c2[idxc2]) and not overlapping: 
            print("start overlapping")
            print("\tc1[idxc1]="+str(c1[idxc1])+", c2[idxc2]="+str(c2[idxc2]))
            #guardem el contour previ (np.concatenate(a, axis=-1))
            #creem nou contour
            new_contour = np.empty((0,1,2), np.int32) #np.array(None)
            #afegim punt a un nou contour
            new_contour = np.concatenate((new_contour, [c1[idxc1]]), axis=0) #c1[idxc1])
            #new_contour = np.vstack((new_contour, [-3, -3])) 
            idxc1 = idxc1 + 1 
            overlapping = True              
        #elif np.array_equal(c1[idxc1], c2[idxc2]) and overlapping:
        #elif np.allclose(c1[idxc1], c2[idxc2], rtol=0.1) and overlapping:
        elif similarPoints(c1[idxc1], c2[idxc2]) and overlapping:
            print("continue overlapping")
            print("\tc1[idxc1]="+str(c1[idxc1])+", c2[idxc2]="+str(c2[idxc2]))
            #afegim punt a un nou contour
            new_contour = np.concatenate((new_contour, [c1[idxc1]]), axis=0)
            idxc1 = idxc1 + 1 
        elif not np.array_equal(c1[idxc1], c2[idxc2]) and overlapping:
            #print("stop overlapping")
            final_contours.append(new_contour)
            new_contour = None
            overlapping = False
            idxc2 = 0
            idxc1 = idxc1 + 1
        #else:
            #print("no coincidence")

        if idxc2 < len(c2)-1:
            idxc2 = idxc2 + 1
        else:
            if new_contour is not None:
                final_contours.append(new_contour)
                new_contour = None
            overlapping = False
            idxc2 = 0
            idxc1 = idxc1 + 1
    if new_contour is not None:
        final_contours.append(new_contour)
    #print(final_contours)    
    return final_contours

def similarPoints(p1, p2):
    #print("p1=", p1)
    if abs(p1[0][0]-p2[0][0])+abs(p1[0][1]-p2[0][1]) <= 185:
        return True
    else:
        return False    
