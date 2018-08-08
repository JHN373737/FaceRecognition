import cv2
import os

def display_img(title, img):
    cv2.imshow(str(title), img)
    print("press any key to close window")
    cv2.waitKey(0)  # wait for x ms -> 0 means wait until a key is pressed
    cv2.destroyAllWindows()

def load_image(image_path):
    if os.path.exists(image_path):
        img = cv2.imread(image_path)  # load image
    else:
        img = None
        print(str(image_path) + " not found bruh")
    return img

#loads, trains opencv detection classifier
def load_detection_classifier(opencv_classifier_path):
    if os.path.exists(opencv_classifier_path):
        classifier = cv2.CascadeClassifier(opencv_classifier_path)
    else:
        classifier = None
        print(str(opencv_classifier_path) + " not found bruhh")
    return classifier