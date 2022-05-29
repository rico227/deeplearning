import cv2 as cv
import glob
import matplotlib.pyplot as plot
from PIL import Image
import datetime

# get images from data path
# img = cv.imread("test/WIN_20220521_14_05_34_Pro.jpg")
images = [cv.imread(file) for file in glob.glob("test/*")]
# convert the images to grayscale
# img_gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
images_gray = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
# using cascade classifier for general face detection
cascade = cv.CascadeClassifier("files/haarcascade_frontalface_default.xml")
# detect faces
# faces = cascade.detectMultiScale(img_gr, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
faces = [cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100)) for img_gray in
         images_gray]
#
i = 0
j = 0
for img in images:
    for x, y, width, height in faces[i]:
        cv.rectangle(img, (x, y), (x + width, y + height), color=(0, 0, 255), thickness=2)
        # crop image at ROI
        crop_image = img[y:y + height, x:x + width]
        saved = cv.imwrite(f"images/me_{j}.jpg", crop_image)
        if saved:
            print(f"saved to images/me_{j}.jpg")
        j += 1
    i += 1
    # show images with detected faces
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
