import cv2
import numpy as np
import sys
import os

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/jump2/imagesFull /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/jump2/small 0 500
#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/jump2/out_opencv /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/jump2/small 0 500

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/running1/imagesFull /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/running1/small 400 600
#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/running1/out_opencv /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/running1/small 400 600

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/man_walk_1/imagesFull /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/man_walk_1/small 250 500
#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/man_walk_1/out_opencv /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/man_walk_1/small 250 500

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/shuffle2/imagesFull /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/shuffle2/small 250 500
#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/shuffle2/out_opencv /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/shuffle2/small 250 500

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/arizona3/imagesFull /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/arizona3/small 250 500
#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/arizona3/out_opencv /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/arizona3/small 250 500

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/fbd5/imagesFull /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/fbd5/small 100 400
#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/fbd5/out_opencv /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/fbd5/small 100 400

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/arizona1/imagesFull /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/arizona1/small 300 400
#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/arizona1/out_opencv /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/arizona1/small 300 400

#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/ruben2/imagesFull /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/ruben2/small 300 400
#python util_crop.py /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/ruben2/out_opencv /Users/rtous/Dropbox/recerca/PAPERS/2024_seg4art/results/ruben2/small 300 400


if __name__ == "__main__":
    
    inputpath = sys.argv[1]
    outputpath = sys.argv[2]
    FROM_X = int(sys.argv[3])
    SIZE = int(sys.argv[4])

    for filename in sorted(os.listdir(inputpath)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
            print("process("+inputpath+"/"+filename+", "+outputpath+")")
            #Read image with opencv
            img = cv2.imread(os.path.join(inputpath, filename), cv2.IMREAD_UNCHANGED)
            assert img is not None, "file could not be read, check with os.path.exists()"
           
            #crop_img = img[0:, FROM_X:FROM_X+SIZE]
            crop_img = img[:, FROM_X:FROM_X+SIZE]

            #b_channel, g_channel, r_channel = cv2.split(crop_img)
            #alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
            #img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
            cv2.imwrite(os.path.join(outputpath, filename), crop_img)
           