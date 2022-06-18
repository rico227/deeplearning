import glob
import os.path
import random
import PIL
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# constants
IMAGES_PATH = "images/"
IMAGES_ME_PATH = IMAGES_PATH + "me/*"
IMAGES_SE_PATH = IMAGES_PATH + "se"
IMAGES_SE_INPUTPATH = IMAGES_SE_PATH + "/*/*/*"
IMG_SIZE = 192
BATCH_SIZE = 16


def showRandomImages(images, amount):
    for image in random.sample(images, amount):
        plt.imshow(image)
        plt.axis("off")
        plt.show()


# get amount of pictures of me
amountImgMe = len([file for file in glob.glob(IMAGES_ME_PATH) if os.path.isfile(file)])
print(amountImgMe)
# get amount of pictures of something else
amountImgSe = len([file for file in glob.glob(IMAGES_SE_PATH) if os.path.isfile(file)])
print(amountImgSe)

# get random faces database in format for tf dataset creating
# if each subdirectory contains images for a class tf assigns labels automatically
# get max pictures of se
imagesSe = [image.load_img(file) for file in glob.glob(IMAGES_SE_INPUTPATH)]
# save every image in subdirectory 'se' using Pillow
# saveImg = [img.save(f"{IMAGES_SE_PATH}/se_{i}.jpg") for i, img in enumerate(random.sample(imagesSe, amountImgMe))]

# create tf datasets
trainDataset = tf.keras.utils.image_dataset_from_directory(
    IMAGES_PATH,
    labels='inferred',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

validationDataset = tf.keras.utils.image_dataset_from_directory(
    IMAGES_PATH,
    labels='inferred',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

# determine how many batches are available in validation dataset
validationBatches = tf.data.experimental.cardinality(validationDataset)
testDataset = validationDataset.take(validationBatches // 5)
validationDataset = validationDataset.skip(validationBatches // 5)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validationDataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(testDataset))

# visualize the data
classNames = trainDataset.class_names
plt.figure(figsize=(10, 10))
for images, labels in trainDataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classNames[labels[i]])
        plt.axis("off")

# to show plot with pycharm
plt.show()

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
trainDataset = trainDataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validationDataset = validationDataset.cache().prefetch(buffer_size=AUTOTUNE)

## Standardize the data
# get RGB values from [0, 255] range to [0, 1] to have small input values for the neural network
rescaleImagesLayer = layers.Rescaling(1. / 255)
# apply rescaling
normalizedDataset = trainDataset.map(lambda x, y: (rescaleImagesLayer(x), y))
imageBatch, labelsBatch = next(iter(normalizedDataset))
# print batch tensor shape for visualization
print(imageBatch.shape)

# apply data augmentation from source: https://www.tensorflow.org/tutorials/images/data_augmentation#overview
dataAugmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# visualize data augmentation
for image, _ in trainDataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = dataAugmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
# to show plot with pycharm
plt.show()

# use transfer learning with tensorflow hub https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
import tensorflow_hub as hub



# use mobilenetv2 for first test
preprocessInput = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)

# Create the base model from the pre-trained model MobileNet V2
IMG_SIZE = (IMG_SIZE, IMG_SIZE)
IMG_SHAPE = IMG_SIZE + (3,)
baseModel = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')

imageBatch, labelBatch = next(iter(trainDataset))
feature_batch = baseModel(imageBatch)
print(feature_batch.shape)

# freeze the convolutional base
baseModel.trainable = False

# show model
baseModel.summary()

#
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = dataAugmentation(inputs)
x = preprocessInput(x)
x = baseModel(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# train model
initial_epochs = 30

loss0, accuracy0 = model.evaluate(validationDataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(trainDataset,
                    epochs=initial_epochs,
                    validation_data=validationDataset)


# learning curves

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Evaluation and prediction
loss, accuracy = model.evaluate(testDataset)
print('Test accuracy :', accuracy)

# Retrieve a batch of images from the test set
image_batch, label_batch = testDataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(classNames[predictions[i]])
  plt.axis("off")

plt.show()