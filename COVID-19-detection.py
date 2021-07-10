# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, Xception
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, sys
from tkinter.filedialog import askopenfilename
from tkinter import Tk     # from tkinter import Tk for Python 3.x

# determining paths
dataset_path = "COVID19_Data"
MODEL = [VGG16]#, VGG19, ResNet50, Xception]
model_name = ["VGG16", "VGG19", "ResNet50", "Xception"]
model_initializer = "glorot_uniform"#,"he_uniform"#'random_normal'#
model_optimizer = "Adam"
# initialize the initial learning rate, number of epochs to train for,
# and batch size
IMG_W = 128
IMG_H = 128
CHANNELS = 3

INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)

INIT_LR = 1e-3
EPOCHS = 3
BS = 8

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]
	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (IMG_W, IMG_H))
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)
    
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

# sys.stdout = open("log.txt", "w")
for i in range(0,len(MODEL)):
    file_name = model_name[i]+"_"+model_initializer+"_"+model_optimizer
    print("*********************************************************************************")
    print(file_name)
    print("*********************************************************************************")    
    file_name = 'fig/'+file_name
    # load the VGG16 network, ensuring the head FC layer sets are left
    # offd
    baseModel = MODEL[i](weights="imagenet", include_top=False,
    	input_tensor=Input(shape=INPUT_SHAPE))
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu", kernel_initializer=model_initializer)(headModel) #glorot_uniform
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # model = models_sch.vgg16()
    # print(model)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
    	layer.trainable = False
        
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
    	metrics=["accuracy"])
    model.summary()
    # train the head of the network
    print("[INFO] training head...")
    H = model.fit_generator(
    	trainAug.flow(trainX, trainY, batch_size=BS),
    	steps_per_epoch=len(trainX) // BS,
    	validation_data=(testX, testY),
    	validation_steps=len(testX) // BS,
    	epochs=EPOCHS)
    
    # serialize the model to disk
    print("[INFO] saving COVID-19 detector model...")
    model.save(file_name+"_model.model", save_format="h5")
    
    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
    	target_names=lb.classes_))  

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    
    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title(model_name[i]+" Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(file_name+"_plot.png")
    
# sys.stdout.close()
print("Please select a CT Scan image ...")
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (IMG_W, IMG_H))
data = []
data.append(image)
data = np.array(data) / 255.0
predIdxs = model.predict(data, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
if predIdxs[0]==1:
    print("Your COVID-19 Test Result is POSITIVE ")
else:    
    print("Your COVID-19 Test Result is NEGATIVE ")
    

    