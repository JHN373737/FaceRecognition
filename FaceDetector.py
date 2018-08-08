import Initializer

import cv2
import os
import copy

# takes image and opencv classifier (Haar or LBP)
# return list of faces detected (coordinates of rectangle around face)
def detect_faces(img, opencv_classifier):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale bc opencv classifiers expect
    # get list of coordinates (rectangle) for all faces
    face_list = opencv_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)  # scalefactor is 1.2 to rescale for faces closer to camera
    #print("#faces found: " + str(len(face_list)))
    return face_list


# takes image and opencv classifier (Haar or LBP)
# returns list of tuples: (grayscale image of face only, coordinates of that face) for each face in image
def get_faces(img, opencv_classifier):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    # loop through detected faces and draw rectangles around them
    for (x, y, w, h) in face_list:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # img to draw on, start coord, end coord, color of rect, line width of rect
    return img_copy





if __name__ == '__main__':
    image_path = "Harry_Potter_Cast.jpg"
    opencv_classifier_path = "opencv/sources/data/lbpcascades/lbpcascade_frontalface_improved.xml"
    img = Initializer.load_image(image_path)
    opencv_classifier = Initializer.load_detection_classifier(opencv_classifier_path)
    #boxed_img = get_boxed_faces(img, opencv_classifier)
    #Initializer.display_img("boxed", boxed_img)
    face_only_list = get_faces(img, opencv_classifier)
    for img, coord in face_only_list:
        print(coord)
        Initializer.display_img("face_only", img)
