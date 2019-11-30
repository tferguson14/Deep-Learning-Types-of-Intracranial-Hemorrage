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

print ('Packages ready!')


# Load Data
# train = pd.read_csv(".././rsna-intracranial-hemorrhage-detection/stage_1_train.csv")
# sub = pd.read_csv(".././rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv")
# train_images = os.listdir(".././rsna-intracranial-hemorrhage-detection/stage_1_train_images/")
# test_images = os.listdir(".././rsna-intracranial-hemorrhage-detection/stage_1_test_images/")

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

train.head()

print ('Train type =', list(train.type.unique()))
print ('Train label =', list(train.Label.unique()))
#train.to_csv('train.csv', index=False)


# Basic Counts
print ('Number of Patients: ', train.PatientID.nunique())

train.type.value_counts()

# Labels
print(train.Label.value_counts())
sns.countplot(x='Label', data=train)

train.groupby('type').Label.value_counts()

sns.countplot(x="Label", hue="type", data=train)




# Visualization
# TRAIN_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"
# TEST_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"
# BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'
# TRAIN_DIR = 'stage_1_train_images/'
# TEST_DIR = 'stage_1_test_images/'

TRAIN_IMG_PATH = ".././rsna-intracranial-hemorrhage-detection/stage_2_train/"
TEST_IMG_PATH = ".././rsna-intracranial-hemorrhage-detection/stage_2_test/"
BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'
TRAIN_DIR = 'stage_1_train_images/'
TEST_DIR = 'stage_1_test_images/'


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
    plt.show()

case = 5
data = pydicom.dcmread(TRAIN_IMG_PATH+train_images[case])

print(data)
window_center , window_width, intercept, slope = get_windowing(data)


#displaying the image
img = pydicom.read_file(TRAIN_IMG_PATH+train_images[case]).pixel_array

img = window_image(img, window_center, window_width, intercept, slope)
plt.imshow(img, cmap=plt.cm.bone)
plt.grid(False)


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

test = pd.DataFrame(sub.filename.unique(), columns=['filename'])
print ('Test:', test.shape[0])
test.head()

np.random.seed(1234)
sample_files = np.random.choice(os.listdir(TRAIN_IMG_PATH), 200000)
sample_df = train[train.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]

pivot_df = sample_df[['Label', 'filename', 'type']].drop_duplicates().pivot(
    index='filename', columns='type', values='Label').reset_index()
print(pivot_df.shape)
pivot_df.head()


def save_and_resize(filenames, load_dir):
    save_dir = '/kaggle/tmp/'
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

save_and_resize(filenames=sample_files, load_dir=BASE_PATH + TRAIN_DIR)
save_and_resize(filenames=os.listdir(BASE_PATH + TEST_DIR), load_dir=BASE_PATH + TEST_DIR)


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
        directory='/kaggle/tmp/',
        x_col='filename',
        class_mode=None,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

def create_flow(datagen, subset):
    return datagen.flow_from_dataframe(
        pivot_df,
        directory='/kaggle/tmp/',
        x_col='filename',
        y_col=['any', 'epidural', 'intraparenchymal',
               'intraventricular', 'subarachnoid', 'subdural'],
        class_mode='other',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        subset=subset
    )

# Using original generator
data_generator = create_datagen()
train_gen = create_flow(data_generator, 'training')
val_gen = create_flow(data_generator, 'validation')
test_gen = create_test_gen()


# DenseNet Model

densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(6, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']
    )

    return model

model = build_model()
model.summary()


# Training

checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

total_steps = sample_files.shape[0] / BATCH_SIZE

history = model.fit_generator(
    train_gen,
    steps_per_epoch=total_steps * 0.85,
    validation_data=val_gen,
    validation_steps=total_steps * 0.15,
    callbacks=[checkpoint],
    epochs=11
)


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()

# Submission

model.load_weights('model.h5')
y_test = model.predict_generator(
    test_gen,
    steps=len(test_gen),
    verbose=1
)


test_df = test.join(pd.DataFrame(y_test, columns = ['any', 'epidural', 'intraparenchymal',
         'intraventricular', 'subarachnoid', 'subdural']))

# Unpivot table
test_df = test_df.melt(id_vars=['filename'])

# Combine the filename column with the variable column
test_df['ID'] = test_df.filename.apply(lambda x: x.replace('.png', '')) + '_' + test_df.variable
test_df['Label'] = test_df['value']

test_df[['ID', 'Label']].to_csv('submission.csv', index=False)

test_df[['ID', 'Label']].head(10)
