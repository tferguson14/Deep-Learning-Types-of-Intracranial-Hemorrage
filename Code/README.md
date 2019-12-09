# Deep Learning Project: Identifying Types of Intracranial Hemorrhage

Intracranial hemorrhage, which is bleeding inside the skeletal structure of the brain, is responsible for approxately 
10% of all strokes in the US. When a hemorrhage is suspected, medical specialists review images of the patient’s cranium 
to determine if there is in fact a hemorrhage and the type of hemorrhage. The process of identifying hemorrhages can be 
complicated and time consuming.  

The main objective of this project is to create an algorithm to correctly classify the presence of hemorrhage as well as 
the type of hemorrhage, if present. The five types of hemorrhages that we aim to correctly classify are intraparenchymal, intraventricular, subarachnoid, subdural, and epidural. 
The dataset were obtained from the Kaggle competition.

###Network Training and Algorithm
For this project we used Keras, which is an open source neural network library, to program and train our image 
classification algorithm. After reviewing some of the available coding examples on Kaggle’s website, as well as several 
research articles related to image classification, we decided to build an ensemble model with the following components: 
a convolutional neural network (CNN) built from scratch, a pretained model called Densenet, and another pretrained model 
called VGG16. From our research, CNN models, Densenet, and VGG16 models were very successful at classifying images 
individually. We thought the best strategy to improve the accuracy of the individual models would be to use a Machine Learning 
strategy called ensemble modeling. Ensembling modeling combines the predictions of two or more models into one prediction. 
It is generally accepted that ensemble models tend to be more accurate than the individual predictions of their component models.
 

Before to execute this project it is necessary to install:

- pip install keras
- pip install pydicom
- pip install opencv-python
- pip install tqdm

Since the data is considerably big, you will require to have access to AWS or GCP.


STEPS TO RUN THE MODEL

1. Set up the environment 
2. Place the raw data in: /home/ubuntu/Machine-Learning/Final-Project-Group9/
3. Go to the directory: cd /home/ubuntu/Machine-Learning/Final-Project-Group9/Code/
4. Execute FinalCode.py

This will run the preprocessing of images, train and predict. 