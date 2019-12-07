import numpy as np
import pandas as pd
import os
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
import keras
from keras import layers
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, MaxPool2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Input, Average, average
from keras.optimizers import Adam
from tqdm import tqdm

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

print ('Number of Patients: ', train.PatientID.nunique())

print(train.type.value_counts())

# Labels
print(train.Label.value_counts())

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


case = 5
data = pydicom.dcmread(TRAIN_IMG_PATH+train_images[case])

print(data)
window_center , window_width, intercept, slope = get_windowing(data)

test = pd.DataFrame(sub.filename.unique(), columns=['filename'])
print ('Test:', test.shape[0])
print(test.head())

np.random.seed(1234)
sample_files = np.random.choice(os.listdir(TRAIN_IMG_PATH), 2000)
np.random.shuffle(sample_files)
print(type(sample_files))
print(sample_files.shape)
sample_train, sample_test = sample_files[:1500], sample_files[1500:]
print(sample_train.shape)
print(sample_test.shape)
sample_df_train = train[train.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_train)]
sample_df_test = train[train.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_test)]


pivot_df_train = sample_df_train[['Label', 'filename', 'type']].drop_duplicates().pivot(
    index='filename', columns='type', values='Label').reset_index()
print(pivot_df_train.shape)
print(pivot_df_train.head())

pivot_df_test = sample_df_test[['Label', 'filename', 'type']].drop_duplicates().pivot(
    index='filename', columns='type', values='Label').reset_index()
print(pivot_df_test.shape)
print(pivot_df_test.head())

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

save_and_resize(filenames=sample_train, load_dir=BASE_PATH + TRAIN_DIR)
save_and_resize(filenames=sample_test, load_dir=BASE_PATH + TRAIN_DIR)
save_and_resize(filenames=os.listdir(BASE_PATH + TEST_DIR), load_dir=BASE_PATH + TEST_DIR)

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

def create_datagen_test():
    return ImageDataGenerator(
        zoom_range=0.1,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images,

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
        pivot_df_train,
        directory='/home/ubuntu/Machine-Learning/Final-Project-Group9/tmp/',
        x_col='filename',
        y_col=['any', 'epidural', 'intraparenchymal',
               'intraventricular', 'subarachnoid', 'subdural'],
        class_mode='raw',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        subset=subset
    )

def create_flow_test(datagen):
    return datagen.flow_from_dataframe(
        pivot_df_test,
        directory='/home/ubuntu/Machine-Learning/Final-Project-Group9/tmp/',
        x_col='filename',
        y_col=['any', 'epidural', 'intraparenchymal',
               'intraventricular', 'subarachnoid', 'subdural'],
        class_mode='raw',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,

    )

# Using original generator
data_generator = create_datagen()
data_generator_test = create_datagen_test()
train_gen = create_flow(data_generator, subset='training')
val_gen = create_flow(data_generator, subset='validation')
test_gen_new = create_flow_test(data_generator_test)
test_gen = create_test_gen()


print(val_gen.labels)
print(val_gen.filenames)
print(val_gen.image_shape)
print(type(val_gen))

print(test_gen_new.image_shape)
print(test_gen_new.samples)
print(test_gen_new.labels)