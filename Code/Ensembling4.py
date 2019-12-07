# https://www.kaggle.com/jesucristo/rsna-introduction-eda-models
#pip install --user  pydicom

# Import Packages
import numpy as np
import pandas as pd
import os
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
print ('Packages ready!')


# Load Data
train = pd.read_csv("/home/ubuntu/Machine-Learning/Final-Project-Group9/rsna-intracranial-hemorrhage-detection/stage_2_train.csv")
sub = pd.read_csv(".././rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv")
train_images = os.listdir(".././rsna-intracranial-hemorrhage-detection/stage_2_train/")
test_images = os.listdir(".././rsna-intracranial-hemorrhage-detection/stage_2_test/")
print ('Train:', train.shape[0])
print ('Sub:', sub.shape[0])

train['type'] = train['ID'].str.split("_", n = 3, expand = True)[2]
train['PatientID'] = train['ID'].str.split("_", n = 3, expand = True)[1]
train['filename'] = train['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

sub['filename'] = sub['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
sub['type'] = sub['ID'].apply(lambda st: st.split('_')[2])

print(train.head())

print ('Train type =', list(train.type.unique()))
print ('Train label =', list(train.Label.unique()))
#train.to_csv('train.csv', index=False)


# Basic Counts
print ('Number of Patients: ', train.PatientID.nunique())

print(train.type.value_counts())

# Labels
print(train.Label.value_counts())
sns.countplot(x='Label', data=train)

print(train.groupby('type').Label.value_counts())

sns.countplot(x="Label", hue="type", data=train)




# Visualization


TRAIN_IMG_PATH = ".././rsna-intracranial-hemorrhage-detection/stage_2_train/"
TEST_IMG_PATH = ".././rsna-intracranial-hemorrhage-detection/stage_2_test/"
BASE_PATH = '/home/ubuntu/Machine-Learning/Final-Project-Group9/rsna-intracranial-hemorrhage-detection/'
TRAIN_DIR = '/stage_2_train/'
TEST_DIR = '/stage_2_test/'


def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)

    return img


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def view_images(images, title='', aug=None):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15, 5))

    for im in range(0, height * width):
        data = pydicom.read_file(os.path.join(TRAIN_IMG_PATH, 'ID_' + images[im] + '.dcm'))
        image = data.pixel_array
        window_center, window_width, intercept, slope = get_windowing(data)
        image_windowed = window_image(image, window_center, window_width, intercept, slope)

        i = im // width
        j = im % width
        axs[i, j].imshow(image_windowed, cmap=plt.cm.bone)
        axs[i, j].axis('off')

    plt.suptitle(title)
    # plt.show()

case = 5
data = pydicom.dcmread(TRAIN_IMG_PATH+train_images[case])

print(data)
window_center , window_width, intercept, slope = get_windowing(data)


#displaying the image
img = pydicom.read_file(TRAIN_IMG_PATH+train_images[case]).pixel_array

img = window_image(img, window_center, window_width, intercept, slope)
# plt.imshow(img, cmap=plt.cm.bone)
# plt.grid(False)


view_images(train[(train['type'] == 'epidural') & (train['Label'] == 1)][:10].PatientID.values, title = 'Images with epidural')

view_images(train[(train['type'] == 'intraparenchymal') & (train['Label'] == 1)][:10].PatientID.values, title = 'Images with intraparenchymal')

view_images(train[(train['type'] == 'subarachnoid') & (train['Label'] == 1)][:10].PatientID.values, title = 'Images with subarachnoid')

view_images(train[(train['type'] == 'subdural') & (train['Label'] == 1)][:10].PatientID.values, title = 'Images with subdural')





# Model
from keras import layers
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm
from keras.applications import VGG16
from keras import applications
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Model, Input
#from keras.engine.topology import Input
from keras.layers import Average
import keras


test = pd.DataFrame(sub.filename.unique(), columns=['filename'])
print ('Test:', test.shape[0])
print(test.head())

np.random.seed(1234)
sample_files = np.random.choice(os.listdir(TRAIN_IMG_PATH), 2000)
sample_df = train[train.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]

pivot_df = sample_df[['Label', 'filename', 'type']].drop_duplicates().pivot(
    index='filename', columns='type', values='Label').reset_index()
print(pivot_df.shape)
print(pivot_df.head())


def save_and_resize(filenames, load_dir):
    save_dir = '/home/ubuntu/Machine-Learning/Final-Project-Group9/tmp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in tqdm(filenames):
        path = load_dir + filename
        new_path = save_dir + filename.replace('.dcm', '.png')

        dcm = pydicom.dcmread(path)
        window_center, window_width, intercept, slope = get_windowing(dcm)
        img = dcm.pixel_array
        img = window_image(img, window_center, window_width, intercept, slope)

        resized = cv2.resize(img, (224, 224))
        res = cv2.imwrite(new_path, resized)
        if not res:
            print('Failed')

# save_and_resize(filenames=sample_files, load_dir=BASE_PATH + TRAIN_DIR)
# save_and_resize(filenames=os.listdir(BASE_PATH + TEST_DIR), load_dir=BASE_PATH + TEST_DIR)


# Data Generator

BATCH_SIZE = 32

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.1,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images,
        validation_split=0.2
    )

def create_test_gen():
    return ImageDataGenerator().flow_from_dataframe(
        test,
        directory='/home/ubuntu/Machine-Learning/Final-Project-Group9/tmp/',
        x_col='filename',
        class_mode=None,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

def create_flow(datagen, subset):
    return datagen.flow_from_dataframe(
        pivot_df,
        directory='/home/ubuntu/Machine-Learning/Final-Project-Group9/tmp/',
        x_col='filename',
        y_col=['any', 'epidural', 'intraparenchymal',
               'intraventricular', 'subarachnoid', 'subdural'],
        class_mode='raw',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        subset=subset
    )

# Using original generator
data_generator = create_datagen()
train_gen = create_flow(data_generator, subset='training')
val_gen = create_flow(data_generator, subset='validation')
test_gen = create_test_gen()


#load data
batch_size = 8
num_epochs = 1
img_rows= 224
img_cols = 224
num_channels = 3
num_classes = 6
total_steps = sample_files.shape[0] / BATCH_SIZE

input_shape = (224, 224, 3)
model_input = Input(shape=input_shape)

keras.backend.clear_session()

vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = vgg16_model.output
x = layers.GlobalAveragePooling2D()(x)
predictions = Dense(6, activation='sigmoid')(x)
vgg16_custom_model = Model(inputs=vgg16_model.input, outputs=predictions, name='vgg16_cnn')

###############################################################################
def compile_and_train(model, num_epochs):

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['acc'])
    filepath = '/home/ubuntu/Machine-Learning/Final-Project-Group9/Code/' + model.name + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='auto', period=1)
    # tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
    history = model.fit_generator(train_gen, steps_per_epoch=total_steps * 0.85, validation_data=val_gen, validation_steps=total_steps * 0.15, callbacks=[checkpoint],epochs=num_epochs)
    # history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_data=(X_valid, Y_valid))
    return history

#compile and train the model
_ = compile_and_train(vgg16_custom_model, num_epochs=num_epochs)
###############################################################################
def evaluate_error(model):
    pred = model.predict_generator(val_gen, steps=len(val_gen), verbose=1)
    # pred = model.predict(X_test, batch_size = batch_size)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, val_gen.labels)) / val_gen.labels.shape[0]

    return error

evaluate_error(vgg16_custom_model)
###############################################################################

import keras
keras.backend.clear_session()

densenet_model = applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = densenet_model.output
x = layers.GlobalAveragePooling2D()(x)
predictions = Dense(6, activation='sigmoid')(x)
densenet_custom_model = Model(inputs=densenet_model.input, outputs=predictions, name='densenet_cnn')

#compile and train the model
_ = compile_and_train(densenet_custom_model, num_epochs=num_epochs)

#Evaluate the model by calculating the error on the test set
evaluate_error(densenet_custom_model)

###############################################################################
densenet_custom_model.load_weights('/home/ubuntu/Machine-Learning/Final-Project-Group9/Code/densenet_cnn.01-0.60.hdf5')
vgg16_custom_model.load_weights('/home/ubuntu/Machine-Learning/Final-Project-Group9/Code/vgg16_cnn.01-0.50.hdf5')

models = [densenet_custom_model, vgg16_custom_model]
#################################################################################


def ensemble(models):
    input_img = Input(shape=(224,224,3))

    outputs = [model(input_img) for model in models] # get the output of model given the input image
    y = Average()(outputs)

    model = Model(inputs=input_img, outputs=y, name='ensemble')
    return model


ensemble_model = ensemble(models)
evaluate_error(ensemble_model)