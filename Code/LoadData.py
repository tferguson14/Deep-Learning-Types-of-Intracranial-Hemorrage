# Inspired by https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/
# https://learning.oreilly.com/library/view/programming-computer-vision/9781449341916/ch01.html
import pathlib
import keras
import numpy as np
import cv2
import glob
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
import pickle
from imutils import paths
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims


class ReadData:
    def __init__(self, dir):
        # os.chdir("/home/ubuntu/Machine-Learning/MidTerm_CristinaGiraldo/Malaria/train") #change directories
        self.data_dir = pathlib.Path(dir)
        self.img_size = 50

    def getLists(self):

        image_count = len(list(self.data_dir.glob('*.png')))

        """Get lists of images path and targets path"""
        # list_img = [file for file in sorted(self.data_dir.glob("*.png"))]
        list_img = list(sorted(paths.list_images(self.data_dir)))
        list_txt = [file for file in sorted(self.data_dir.glob("*.txt"))]
        labels = []

        """Read label in the txt"""
        for f in sorted(self.data_dir.glob("*.txt")):
            with open(f, 'r') as file:
                data = file.read()
                labels.append(data)

        """creates a dataframe with img path, txt path, and label"""
        ziplist = list(zip(list_img, list_txt, labels))
        df = pd.DataFrame(ziplist, columns=['img', 'txt', 'label'], )
        return image_count, list_img, list_txt, df

    def encode(self, df):
        """Encode classes with integers: red blood cell: 0, ring: 1, schizont: 2, trophozoite: 3"""
        class_names = df['label']
        values = np.array(class_names)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        df["target"] = integer_encoded
        return df.drop(["txt", "label"], axis=1) #returns image path and target

    def downloadTrainSet(self, df):
        return df.to_csv(r'/home/ubuntu/Machine-Learning/MidTerm_CristinaGiraldo/Malaria/train/train.csv', index=False)

    def create_training_data(self, df):
        """Converts img to arrays and it is saved along with the target"""
        training_data = []
        df_full = []
        for img, target in zip(df.img, df.target):
            img = os.path.basename(os.path.normpath(img))
            img_array = cv2.imread(os.path.join(self.data_dir, img), cv2.IMREAD_GRAYSCALE)  # convert to array in gray color
            new_img = cv2.resize(img_array, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            # new_img = cv2.equalizeHist(new_img)
            # new_img = cv2.blur(new_img,(5,5))
            new_img = cv2.bilateralFilter(new_img, 9, 75, 75) #blurr
            # new_img = cv2.Laplacian(new_img, cv2.CV_64F)#gradient
            training_data.append([new_img, target])
            df_full.append([img, new_img, target])
        """I shuffled to avoid predicting the same thing all the time"""
        random.shuffle(training_data)
        for sample in training_data:
            print(sample[1])
        return training_data, pd.DataFrame(df_full, columns=['img', 'img_array', 'target'])#pd.DataFrame(training_data)

    def splitData(self, training_data):
        """Split the data and reshape it"""
        X = []
        y = []
        for features, label in training_data:
            X.append(features)
            y.append(label)
        X= np.array(X).reshape(-1, self.img_size, self.img_size)
        return X,y

    def saveData(self, X,y): #to avoid calculating all the time
        pickle_out = open("X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()




dir = "/home/ubuntu/Machine-Learning/MidTerm_CristinaGiraldo/Malaria/train"
malaria_data = ReadData(dir)
img_count, listImg, listTxt, df = malaria_data.getLists()
df_encoded = malaria_data.encode(df)
training_data, df_train = malaria_data.create_training_data(df_encoded)
malaria_data.downloadTrainSet(df_train)
X,y=malaria_data.splitData(training_data) #check this
print("INFO: Saving dataset")
malaria_data.saveData(X, y) #check this
print("fin")




# borrar archivos que ya no me sirven
# import os
# os.remove('data.h5')


