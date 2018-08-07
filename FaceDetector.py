import cv2
import os

class FaceDetector(object):

    # loads image, trains classifier
    def __init__(self, image_path, opencv_classifier_path):
        if os.path.exists(image_path):
            self.img = cv2.imread(image_path)  # load image
        else:
            self.img = None
            print("image not found bruh")

        if os.path.exists(opencv_classifier_path):
            self.classifier = cv2.CascadeClassifier(opencv_classifier_path)
        else:
            self.classifier = None
            print("opencv classifier not found bruhh")


    def display_img(self, title, img):
        cv2.imshow(str(title), img)
        print("press any key to close window")
        cv2.waitKey(0) # wait for x ms -> 0 means wait until a key is pressed
        cv2.destroyAllWindows()


    # takes image and opencv classifier (Haar or LBP)
    # return list of faces detected (coordinates of rectangle around face)
    def detect_faces(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)  # convert to grayscale bc opencv classifiers expect
        # get list of coordinates (rectangle) for all faces
        face_list = self.classifier.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)  # scalefactor is 1.2 to rescale for faces closer to camera
        #print("#faces found: " + str(len(face_list)))
        return face_list


    # takes image and opencv classifier (Haar or LBP)
    # returns list of tuples: (image of face only, coordinates of that face) for each face in image
    def get_faces(self):
        face_list = self.detect_faces()
        ret_list = []
        #loop through detected faces
        for (x,y,w,h) in face_list:
            face_only = self.img[y:y+w, x:x+h]
            face_coord = (x,y,w,h)
            ret_list.append((face_only, face_coord))
        return ret_list


    # takes image and opencv classifier (Haar or LBP)
    # returns image with rectangle drawn around each detected face
    def get_boxed_faces(self):
        img_copy = self.img
        face_list = self.detect_faces()
        # loop through detected faces and draw rectangles around them
        for (x, y, w, h) in face_list:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # img to draw on, start coord, end coord, color of rect, line width of rect
        return img_copy





if __name__ == '__main__':
    image_path = "opencv_stuff/samples/data/lena.jpg"
    opencv_classifier_path = "opencv_stuff/lbpcascades/lbpcascade_frontalface_improved.xml"
    f = FaceDetector(image_path,opencv_classifier_path)
    #boxed_img = f.get_boxed_faces()
    #f.display_img("boxed", boxed_img)
    face_only_list = f.get_faces()
    for img, coord in face_only_list:
        f.display_img("face_only", img)
        print(coord)
