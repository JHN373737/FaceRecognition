import Initializer, FaceDetector, FaceRecognizer
import cv2
import os
import copy
import imghdr
import re
import numpy as np
from pathlib import Path

'''
Used to parse frames from video and retrieve info about faces detected/recognized
    from_video_detection: Used to retrieve frames to use as training data
    from_video_recognition: Used to evaluate performance on videos (return all frames)
'''


# retrieves every xth frame that has at least one face detected and saves to output
# does not edit frames
def from_video_detection(video_path, output_path, opencv_classifier):
    cap = cv2.VideoCapture(video_path)

    count = 0
    while (cap.isOpened()):
        ret_code, frame = cap.read()
        if ret_code == True:
            ret_frame_list = FaceDetector.detect_faces(frame, opencv_classifier)
            if len(ret_frame_list)<=0:
                continue
            if count%11==0:
                write_path = output_path+str(count)+".jpg"
                cv2.imwrite(write_path, frame)
                print(count)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):  # wait key is time(ms) between frames, press q to exit
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()



# retrieves every xth frame that has at least one face detected and recognition confidence>45% and saves to output
# edits frame to have box drawn around recognized faces
def from_video_recognition(video_path, output_path, name_list, detection_classifier, trained_recognizer):
    cap = cv2.VideoCapture(video_path)

    count = 0
    while (cap.isOpened()):
        ret_code, frame = cap.read()
        if ret_code == True:
            ret_frame, skip_code = FaceRecognizer.get_recognition(frame, name_list, detection_classifier, trained_recognizer)

            if skip_code==None: continue
            if count%1==0:
                write_path = output_path+str(count)+".jpg"
                cv2.imwrite(write_path, ret_frame)
                print(count)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):  # wait key is time(ms) between frames, press q to exit
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

#main for recognition

video_path = "vid_test_data/HP_troll_trim_end.mp4"
output_path = "vid_train_data/HP_troll_start2/"
opencv_classifier_path = "opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"
opencv_classifier = Initializer.load_detection_classifier(opencv_classifier_path)

#from_video_detection(video_path,output_path,opencv_classifier)

training_data_path = "vid_train_data/HP_troll_start2/HP_troll_start/"
name_list = FaceRecognizer.get_name_list(training_data_path)
face_list, label_list = FaceRecognizer.preprocess(training_data_path, opencv_classifier)
trained_recognizer = FaceRecognizer.train_recognizer(face_list, label_list, opencv_recognizer_type="LBPH")

from_video_recognition(video_path,output_path,name_list,opencv_classifier,trained_recognizer)
