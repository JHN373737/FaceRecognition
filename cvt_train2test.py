import os
import imghdr
import shutil
from pathlib import Path


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


convert(2, "", "")
print("done")
