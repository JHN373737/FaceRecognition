import FaceDetector, Initializer
import cv2
import os
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
                    face_tuple = FaceDetector.get_faces(img, detection_classifier)[0] # get 1st face and coordinates
                    if face_tuple[0] is not None:
                        face_list.append(face_tuple)
                        label_list.append(label)
                    else:
                        print("No face was detected in "+str(image_path)+ " so the image is not used")
            label += 1

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
        recognizer = cv2.face.createLBPHFaceRecognizer()
    if opencv_recognizer_type == "Eigen":
        recognizer = cv2.face.createEigenFaceRecognizer()
    if opencv_recognizer_type == "Fisher":
        recognizer = cv2.face.createFisherFaceRecognizer()

    recognizer.train(face_list, np.array(label_list))

# takes image, list of names that correspond to labels, detection classifier obj, trained recognizer obj
# if that image is recognized the face will be boxed and the name of the person that the machine thinks it is will appear with the confidence
# returns image with boxed face, labeled name, labeled confidence
def get_recognition(img, training_data_path, detection_classifier, opencv_recognizer_type):
    img_copy = img.copy()
    face, coord = FaceDetector.get_faces(img, detection_classifier)[0] # get 1st face in img

    name_list = get_name_list(training_data_path)
    face_list, label_list = preprocess(training_data_path, detection_classifier)
    trained_recognizer = train_recognizer(face_list, label_list, opencv_recognizer_type)
    label, confidence = trained_recognizer.predict(face)

    face_name = name_list[label]
    label_image(img_copy, coord, face_name, confidence)
    return img_copy


def label_image(img, coord, name, confidence):
    (x,y,w,h) = coord
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    txt = (name + ", " + confidence)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)

if __name__ == '__main__':
    image_path = ""
    detection_classifier_path = ""
    training_data_path = ""
    img = Initializer.load_image(image_path)
    detection_classifier = Initializer.load_classifier(detection_classifier_path)

    ret_img = get_recognition(img,training_data_path, detection_classifier, opencv_recognizer_type="LBPH")
