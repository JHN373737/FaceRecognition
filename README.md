# FaceRecognition
Python project using opencv to detect and recognize faces in both images and videos
	

Created with help of RAMIZ RAJA's article, "FACE RECOGNITION USING OPENCV AND PYTHON: A BEGINNER’S GUIDE," on superdatascience.com



** Note: confidence levels for opencv recognizers are reported as distances between test image and "closest" found training image
	I.E: a distance of 0 represents a confidence of 100%
	-> to determine a threshold (max) distance an image from att_faces and an image of Morty from "Ricky and Morty" TV show were compared
	-> threshold determined to be about 200
	-> Recognition Confidence Percentage = (200-Distance)/200 * 100%

** Note: Package "opencv-contrib-python" must be installed for opencv3 to build fully **

