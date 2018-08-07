import cv2
import os


# takes image and opencv classifier (Haar or LBP)
# returns image with rectangle drawn around each detected face
def get_faces(img, classifier):
    img_copy = img
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert to grayscale bc opencv classifiers expect
    # get list of coordinates (rectangle) for all faces
    face_list = classifier.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5) #scalefactor is 1.2 to rescale for faces closer to camera
    print("#faces found: " + str(len(face_list)))

    #loop through detected faces and draw rectangles around them
    for (x,y,w,h) in face_list:
        cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0,0,255), 2) #img to draw on, start coord, end coord, color of rect, line width of rect
    return img_copy


def display_img(title, img):
    cv2.imshow(str(title), img)
    print("press any key to close window")
    cv2.waitKey(0) # wait for x ms -> 0 means wait until a key is pressed
    cv2.destroyAllWindows()


def run_detection(image_path, opencv_classifier_path):
    img = ""
    ret_img = ""
    if os.path.exists(image_path):
        img = cv2.imread(image_path) # load image
        #display_img("title", img)
    else: print("image not found bruh")

    if os.path.exists(opencv_classifier_path):
        # load lbp classifier's training data for frontal face
        lbp_cascade = cv2.CascadeClassifier(opencv_classifier_path)
        ret_img = get_faces(img, lbp_cascade)

        display_img("detected faces", ret_img)
    else: print("opencv classifier not found bruhh")

    return ret_img



if __name__ == '__main__':
    image_path = "opencv_stuff/samples/data/lena.jpg"
    opencv_classifier_path = "opencv_stuff/lbpcascades/lbpcascade_frontalface_improved.xml"
    run_detection(image_path, opencv_classifier_path)
