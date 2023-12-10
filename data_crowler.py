import requests
import os

TARGET_DIR_TRAIN = "data/fashion/train"
if not os.path.exists(TARGET_DIR_TRAIN):
   os.makedirs(TARGET_DIR_TRAIN)
TARGET_DIR_TEST = "data/fashion/test"
if not os.path.exists(TARGET_DIR_TEST):
   os.makedirs(TARGET_DIR_TEST)

f_train = open('fashion_train.txt', "r")
f_test = open('fashion_test.txt', "r")

train_files = f_train.readlines()
test_files = f_test.readlines()

#os.mkdir('train')
#os.mkdir('test')

for video_url in train_files:
    r = requests.get(video_url[:-1])
    file_name = video_url[:-1].split("/")[-1]
    with open(TARGET_DIR_TRAIN+"/"+file_name,'wb') as f:
        f.write(r.content)

for video_url in test_files:
    r = requests.get(video_url[:-1])
    file_name = video_url[:-1].split("/")[-1]
    with open(TARGET_DIR_TEST+"/"+file_name,'wb') as f:
        f.write(r.content)
