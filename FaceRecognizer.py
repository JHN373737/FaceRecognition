import FaceDetector, Initializer
import cv2
import os
import copy
import imghdr
import re
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
                if imghdr.what(image_path)!= None: #check to make sure it's an image
                    img = Initializer.load_image(image_path)
                    face_tuple = FaceDetector.get_faces(img, detection_classifier)
                    if len(face_tuple)== 1: # if one face detected, keep in training data, else skip
                        face_tuple = face_tuple[0] 
                    else:
                        print("No faces or more than one face found in "+ str(image_path) + " so not used")
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
        print("No faces detected in test image -> skipped")
        return img_copy
            #  what happens if face isn't in training data

    for face, coord in data: # perform recognition on every detected face
        label, confidence = trained_recognizer.predict(face)
        face_name = name_list[label]
        label_image(img_copy, coord, face_name, confidence)
    return img_copy

# does not return image - just predicted face_name and confidence
    # Only works for test images with only one detected face or uses 1st face detected
def get_recognition_stats(img, name_list, detection_classifier, trained_recognizer):
    data = FaceDetector.get_faces(img, detection_classifier) # get all faces/coords in image and try to detect all
    if len(data) <= 0:
        print("No faces detected in test image -> skipped")
        return (None,None)
            #  what happens if face isn't in training data

    face, coord = data[0] #take 1st detected face
    label, confidence = trained_recognizer.predict(face)
    face_name = name_list[label]
    return (face_name, confidence)


def label_image(img, coord, name, confidence):
    (x,y,w,h) = coord
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    txt = (name + ", " + "{0:.3f}".format(confidence))
    cv2.putText(img, txt, (x-10, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)

    
def video_get_recognition(video_path, name_list, detection_classifier, trained_recognizer):
    cap = cv2.VideoCapture(video_path)

    while (cap.isOpened()):
        ret_code, frame = cap.read()
        ret_frame = get_recognition(frame, name_list, detection_classifier, trained_recognizer) #for every frame, perform recognition and return frame

        cv2.imshow('frame', ret_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): # wait key is time(ms) between frames, press q to exit
            break

    cap.release()
    cv2.destroyAllWindows()    
    
    
    
if __name__ == '__main__':
    training_data_path = "YALE_normalized_faces/faces/training_data/"
    test_data_path = "YALE_normalized_faces/faces/test_data/"
    #detection_classifier_path = "opencv/sources/data/lbpcascades/lbpcascade_frontalface_improved.xml"
    detection_classifier_path = "opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"
    detection_classifier = Initializer.load_detection_classifier(detection_classifier_path)

    name_list = get_name_list(training_data_path)

    face_list, label_list = preprocess(training_data_path, detection_classifier)
    trained_recognizer = train_recognizer(face_list, label_list, opencv_recognizer_type="LBPH")

    correct = 0
    incorrect = 0
    total_confidence=0
    incorrect_list = []
    for image_name in os.listdir(test_data_path):
        image_path = Path(test_data_path, image_name)
        img = Initializer.load_image(image_path)
        predicted_name, confidence = get_recognition_stats(img, name_list, detection_classifier, trained_recognizer)
        if predicted_name==None or confidence==None:
            print("no faces detected in "+ image_name)
            continue
        actual_name = re.sub(r'[0-9]+', '', image_name) # remove digits
        actual_name = actual_name.split(".")[0] #get everything before . from file extension

        if actual_name == predicted_name:
            print("correct: "+predicted_name + " correctly predicted with "+ str(confidence) + " confidence")
            correct+=1
            total_confidence+=confidence
        else:
            print("incorrect: "+actual_name+" predicted as "+ predicted_name)
            incorrect_list.append((actual_name,predicted_name))
            incorrect+=1
        #Initializer.display_img("Recognized Faces", ret_img)

    total = len(os.listdir(test_data_path))
    discarded = total-(correct+incorrect)
    print("total: " + str(total))
    print("discarded: " + str(discarded)) #no faces detected so couldn't use
    print("correct: " + str(correct))
    print("incorrect: " + str(incorrect))
    print("incorrect ones are(actual,predicted): " + str(incorrect_list))

    for actual,predicted in incorrect_list: #actual, predicted are also dir names
        actual_path = Path(training_data_path, actual)
        actual_path = Path(actual_path, os.listdir(str(actual_path))[0]) # path of 1st image
        predicted_path = Path(training_data_path, predicted)
        predicted_path = Path(predicted_path, os.listdir(str(predicted_path))[0]) # path of 1st image

        actual_img = Initializer.load_image(actual_path)
        predicted_img = Initializer.load_image(predicted_path)
        Initializer.display_img("actual", actual_img)
        Initializer.display_img("predicted", predicted_img)


    print("After preprocess Detection Performance: "+ "{0:.2f}".format(((total-discarded)/total)*100.0) +"%  ("+ str(total-discarded)+"/"+str(total)+")" )
    print("Recognition Performance: " + "{0:.2f}".format((correct/total)*100.0) +"% correctly recognized" +
          ", with avg confidence: "+ "{0:.3f}".format((total_confidence/total)))



