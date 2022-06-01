import glob
import os.path
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# constants
IMAGES_ME_PATH = "images/me/*"
IMAGES_SE_PATH = "images/se/*/*/*"
SPLIT_DATA = 0.5
IMG_SIZE = 200

def showRandomImages(images, amount):
    for image in random.sample(images, amount):
        plt.imshow(image)
        plt.axis("off")
        plt.show()

####################################
# Maybe create custom tf dataset
####################################

# get amount of pictures of me
amountImgMe = len([file for file in glob.glob(IMAGES_ME_PATH) if os.path.isfile(file)])
# get amount of pictures of something else
amountImgSe = len([file for file in glob.glob(IMAGES_SE_PATH) if os.path.isfile(file)])

# get max pictures of me
imagesMe = [image.load_img(file) for file in glob.glob(IMAGES_ME_PATH)]

# resizing and rescaling images
resizeAndRescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])

imagesMeResizedAndRescaled = [resizeAndRescale(img) for img in imagesMe]

# apply data augmentation from source: https://www.tensorflow.org/tutorials/images/data_augmentation#overview
dataAugmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
])

imagesMeAugmentated = [dataAugmentation(img) for img in imagesMeResizedAndRescaled]

# show a random selection of pictures to verify the correct pictures are used
showRandomImages(imagesMeAugmentated, 3)

# get max pictures of se
imagesSe = [image.load_img(file) for file in glob.glob(IMAGES_SE_PATH)]
showRandomImages(imagesSe, 3)
