import os
import imghdr
import shutil
from pathlib import Path

'''
Used to move a user-inputted number or percentage of images from training folder to test folder
    Use when you have a large data set and need to split into train, test
'''

# mv "num" images from every dir in training data dir to test data dir - renamed to dir_name+a number
def convert(num, train_path, test_path):
    count = 0
    for dir_name in os.listdir(train_path):
        dir_path = Path(train_path, dir_name)
        if os.path.isdir(str(dir_path)):
            file_list = os.listdir(str(dir_path))
            for i in range(num):
                if i<len(file_list):
                    file_path = Path(dir_path, file_list[i])
                    if imghdr.what(str(file_path))!= None:
                        filename, file_extension = os.path.splitext(file_path)
                        new_name = dir_name + str(count)+file_extension
                        dest = Path(test_path, new_name)
                        shutil.move(str(file_path), str(dest))
                        count+=1


#convert(2, "", "")
#print("done")

# mv test_percent/100 * total number of images in train_path to test_path - renamed to dir_name+ number
def convert_percent(test_percent, train_path, test_path):
    count = 0  # added to new name of image to prevent same name errors
    dir_size = 0  # number of images in dir
    for dir_name in os.listdir(train_path):
        dir_path = Path(train_path, dir_name)
        if os.path.isdir(str(dir_path)):
            file_list = os.listdir(str(dir_path))
            dir_size = 0
            for file_name in file_list:
                if imghdr.what(str(Path(dir_path,file_name)))!= None:
                    dir_size+=1
            print(dir_name)
            print(dir_size)

            num = (int)((test_percent/100.0) * dir_size)
            print(num)
            for i in range(num):
                if i < len(file_list):
                    file_path = Path(dir_path, file_list[i])
                    if imghdr.what(str(file_path)) != None:
                        filename, file_extension = os.path.splitext(file_path)
                        new_name = dir_name + str(count) + file_extension
                        dest = Path(test_path, new_name)
                        shutil.move(str(file_path), str(dest))
                        count += 1

#convert_percent(10, "ms_celeb_data/training_data", "ms_celeb_data/test_data/")
#print("done")
