import FaceDetector, Initializer
import cv2
import os
import copy
import numpy as np
from pathlib import Path


# takes the detection classfier and path to the root directory of training data
  # assumes every distinct sub-folder will contain images of only one person -> 1st detected face is used
  # only checks one level below root -> if a sub-folder has a sub-folder, the 2nd will not be checked
    # root
    #   - dir1
    #       - img of Harry Potter
    #       - img of Harry Potter and other people that won't be used
    #       - directory that won't be checked
    #   - dir2
    #       - img of not Harry Potter
    #       - another img of not Harry Potter

# preprocess will detect faces and create two lists with same indexing -> list of (detected faces, coords) and list of labels
    # face at index i will have label at index i
    # dir number will be label (opencv requires integer labels) -> first dir = 0, 2nd dir = 1
def preprocess(training_data_path, detection_classifier):
    face_list = []
    label_list = []
    dirs = os.listdir(training_data_path) # get names of everything one level below training_data_path
    label = 0
    for dir in dirs:
        dir_path = Path(training_data_path, dir)
        if os.path.isdir(dir_path): # only keep directories
            images = os.listdir(str(dir_path))
            for image in images:
                image_path = Path(dir_path, image)
                if ( (not os.path.isdir(image_path)) and (not image.startswith(".")) ): # assume it is an image if not dir and doesn't start with '.' (system file)
                    img = Initializer.load_image(image_path)
                    face_tuple = FaceDetector.get_faces(img, detection_classifier)
                    if len(face_tuple)> 0:
                        face_tuple = face_tuple[0]
                    else:
                        print("No faces found in "+ str(image_path))
                        continue
                    if face_tuple[0] is not None: # face_tuple[0] = face image in grayscale
                        face_list.append(face_tuple[0])
                        label_list.append(label)
                    else:
                        print("No face was detected in "+str(image_path)+ " so the image is not used")
            label += 1
    if len(face_list)<= 0:
        print("No faces detected in training data -> cannot proceed")
        raise SystemExit

    return face_list, label_list

# names are accessed through labels -> name[label] = name of person with that label
def get_name_list(training_data_path):
    name_list = []
    dirs = os.listdir(training_data_path)  # get names of everything one level below training_data_path
    for dir in dirs:
        dir_path = Path(training_data_path, dir)
        if os.path.isdir(dir_path):  # only keep directories
            name_list.append(dir) #name of dir will be the name of the person

    return name_list

def train_recognizer(face_list, label_list, opencv_recognizer_type="LBPH"):
    if opencv_recognizer_type == "LBPH":
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    if opencv_recognizer_type == "Eigen":
        recognizer = cv2.face.EigenFaceRecognizer_create()
    if opencv_recognizer_type == "Fisher":
        recognizer = cv2.face.FisherFaceRecognizer_create()

    recognizer.train(face_list, np.array(label_list)) #face_list[0] gives face images
    return recognizer

# takes image, list of names respective to labels (name[label] = person with that label), detection classifier object, trained recognizer object
# boxes all faces recognized in image and labels the box with name and confidence (distance to most similar image in training data) of prediction
# returns image with boxed faces, labeled names, labeled confidences
def get_recognition(img, name_list, detection_classifier, trained_recognizer):
    img_copy = img.copy()
    data = FaceDetector.get_faces(img, detection_classifier) # get all faces/coords in image and try to detect all
    if len(data) <= 0:
        print("No faces recognized in test image -> cannot proceed")
        raise SystemExit
            #  what happens if face isn't in training data

    for face, coord in data: # perform recognition on every detected face
        label, confidence = trained_recognizer.predict(face)
        face_name = name_list[label]
        label_image(img_copy, coord, face_name, confidence)
    return img_copy


def label_image(img, coord, name, confidence):
    (x,y,w,h) = coord
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    txt = (name + ", " + "{0:.3f}".format(confidence))
    cv2.putText(img, txt, (x-10, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)

if __name__ == '__main__':
    image_path = "test-data/test2.jpeg"
    detection_classifier_path = "opencv/sources/data/lbpcascades/lbpcascade_frontalface_improved.xml"
    training_data_path = "training-data"
    img = Initializer.load_image(image_path)
    detection_classifier = Initializer.load_detection_classifier(detection_classifier_path)

    name_list = get_name_list(training_data_path)

    face_list, label_list = preprocess(training_data_path, detection_classifier)
    trained_recognizer = train_recognizer(face_list, label_list, opencv_recognizer_type="LBPH")

    ret_img = get_recognition(img, name_list, detection_classifier, trained_recognizer)
    Initializer.display_img("Recognized Faces", ret_img)
