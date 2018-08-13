import Initializer

import cv2
import os
import copy
from pathlib import Path

# takes image and opencv classifier (Haar or LBP)
# return list of faces detected (coordinates of rectangle around face)
def detect_faces(img, opencv_classifier):
    gray_img = Initializer.cvt2GRAY(img) # convert to grayscale bc opencv classifiers expect
    # get list of coordinates (rectangle) for all faces
    #face_list = opencv_classifier.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3)  # scalefactor is 1.2 to rescale for faces closer to camera
    face_list = opencv_classifier.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=3)
    #print("#faces found: " + str(len(face_list)))
    return face_list


# takes image and opencv classifier (Haar or LBP)
# returns list of tuples: (grayscale image of face only, coordinates of that face) for each face in image
def get_faces(img, opencv_classifier):
    gray_image = Initializer.cvt2GRAY(img)
    face_list = detect_faces(img, opencv_classifier)
    ret_list = []
    #loop through detected faces
    for (x,y,w,h) in face_list:
        face_only = gray_image[y:y+w, x:x+h]
        face_coord = (x,y,w,h)
        ret_list.append((face_only, face_coord))
    return ret_list


# takes image and opencv classifier (Haar or LBP)
# returns image with rectangle drawn around each detected face
def get_boxed_faces(img, opencv_classifier):
    img_copy = img.copy()
    face_list = detect_faces(img, opencv_classifier)
    #bad_img = Initializer.load_image("trashcan.jpg")
    if len(face_list)<=0:
        return img_copy
    # loop through detected faces and draw rectangles around them
    for (x, y, w, h) in face_list:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # img to draw on, start coord, end coord, color of rect, line width of rect
    return img_copy


def video_get_boxed_faces(video_path, opencv_classifier, output_path):
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # codec is platform dependent - windows: DIVX
    out = cv2.VideoWriter(output_path, fourcc, 29.88, (640, 294))  # 29.88 fps, 640 width, 294 height

    while (cap.isOpened()):
        ret_code, frame = cap.read()
        if ret_code == True:
            ret_frame = get_boxed_faces(frame, opencv_classifier)  # for every frame, draw box around faces and return frame

            out.write(ret_frame) # write the new frame
            cv2.imshow('frame', ret_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # wait key is time(ms) between frames, press q to exit
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    image_path = "HP_test_detect.jpg"
    opencv_classifier_path = "opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"
    img = Initializer.load_image(image_path)
    #Initializer.display_img("title",img)a
    opencv_classifier = Initializer.load_detection_classifier(opencv_classifier_path)

    video_path = "vid_test_data/HP_troll_trim_start2.mp4"
    output_path = "vid_test_data/HP_troll_trim_start2_out.mp4"
    boxed_img = get_boxed_faces(img,opencv_classifier)
    face_only_list = get_faces(img, opencv_classifier)


    #Initializer.display_img("boxed", boxed_img)


    #for img, coord in face_only_list:
    #    print(coord)
    #    Initializer.display_img("face_only", img)


    #video_get_boxed_faces(video_path, opencv_classifier, output_path)
