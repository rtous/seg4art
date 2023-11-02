import shutil
import os
import sys

#Use this if you merge multiple scenes parts (to make the video you need consecutive numbers)
#Put the original pngs into out_opencv_all
#Run python rename.py man_walk_1

if __name__ == "__main__":
    SCENE_NAME = sys.argv[1]
    inputpath = "/Users/rtous/DockerVolume/seg4art/data/scenes/"+SCENE_NAME+"/out_opencv_all"
    outputpath = "/Users/rtous/DockerVolume/seg4art/data/scenes/"+SCENE_NAME+"/out_opencv"
    if not os.path.exists(outputpath):
       os.makedirs(outputpath)
    for i, filename in enumerate(sorted(os.listdir(inputpath))):
        shutil.copy(os.path.join(inputpath, filename), os.path.join(outputpath, "%03d.png"%i)) 