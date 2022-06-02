import cv2 as cv
import glob
import tensorflow as tf
import matplotlib.pyplot as plot
from PIL import Image
import datetime

# get images from data path
# img = cv.imread("test/WIN_20220521_14_05_34_Pro.jpg")
images = [cv.imread(file) for file in glob.glob(r"C:\repos\deeplearning\debug_proj\images\mePreDetection\*")]
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
k = 0
for img in images:
    for x, y, width, height in faces[i]:
        # print rectangle around ROI for debugging
        #cv.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 1)
        # crop image at ROI
        crop_image = img[y:y + height, x:x + width]
        # show image for manual validation
        cv.imshow('cropped image', crop_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # ask user to sort image
        print("Press 'm' if the picture shows your face. Press 's' if the picture shows something or somebody else. "
              "Press 'q' if you don't want to save the picture")
        decision = input()
        if decision == 'm':
            saved = cv.imwrite(f"images/me/me_{j}.jpg", crop_image)
            if saved:
                print(f"saved to images/me/me_{j}.jpg")
            j += 1
        elif decision == 's':
            saved = cv.imwrite(f"images/se/se_{k}.jpg", crop_image)
            if saved:
                print(f"saved to images/se/se_{k}.jpg")
            k += 1
    i += 1
    # show images with detected faces
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
