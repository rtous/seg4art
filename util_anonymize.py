import cv2
import numpy as np
import sys
import os

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/jump2/small /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/jump2/anonymized 50 100 50 100

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/running1/small /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/running1/anonymized 400 600

#python util_anonymize.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/man_walk_1/small /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/man_walk_1/anonymized 100 200 100 200


if __name__ == "__main__":
    
    inputpath = sys.argv[1]
    outputpath = sys.argv[2]
    x1 = int(sys.argv[3])
    x2 = int(sys.argv[4])
    y1 = int(sys.argv[5])
    y2 = int(sys.argv[6])

    for filename in sorted(os.listdir(inputpath)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
            print("process("+inputpath+"/"+filename+", "+outputpath+")")
            #Read image with opencv
            img = cv2.imread(os.path.join(inputpath, filename), cv2.IMREAD_UNCHANGED)
            assert img is not None, "file could not be read, check with os.path.exists()"
           
            #crop_img = img[0:, FROM_X:FROM_X+SIZE]
            #crop_img = img[:, FROM_X:FROM_X+SIZE]
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,0), thickness=-1)

            #b_channel, g_channel, r_channel = cv2.split(crop_img)
            #alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
            #img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
            cv2.imwrite(os.path.join(outputpath, filename), img)
           