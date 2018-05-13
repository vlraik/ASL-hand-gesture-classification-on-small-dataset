
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os,sys
import shutil
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.optimizers import *

#from IPython.display import Image
from keras import applications


# Here, we try to resize all the images to 100x100 pixel size, we aren't reducing the size since that might leads to information loss.

# In[2]:


path = ("handgesturedata/")
if not os.path.exists("preview"):
    os.makedirs("preview")
img_width, img_height = 100, 100
batchsize=32
for subfolder in os.listdir(path):
    print(subfolder)
    for item in os.listdir(path+subfolder):
        #print(item)
        if item.endswith(".bmp"):
            im = Image.open(path+subfolder+"/"+item)
            print(path+subfolder+"/"+item)
            f, e = os.path.split(path+subfolder+"/"+item)
            imResize = im.resize((100,100), Image.ANTIALIAS)
            im.close()
            imResize.save(path+subfolder+"/"+item,"JPEG")


# Now we divide the dataset into training and validation set for training a convolutional neural network

# In[3]:


for subfolder in os.listdir(path):
    print(subfolder)
    trainingset = os.listdir(path+subfolder)[:2*len(os.listdir(path+subfolder))//3]
    validationset = os.listdir(path+subfolder)[2*len(os.listdir(path+subfolder))//3:]
    print(trainingset)
    print(validationset)
    if not os.path.exists("data/train/"+subfolder):
        os.makedirs("data/train/"+subfolder)
        os.makedirs("data/validation/"+subfolder)
        
    for item in trainingset:
        shutil.copy2(path+subfolder+'/'+item,"data/train/"+subfolder)
        
    for item in validationset:
        shutil.copy2(path+subfolder+'/'+item,"data/validation/"+subfolder)


# In[4]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
train_generator = ImageDataGenerator(width_shift_range = 0.1,height_shift_range = 0.1, horizontal_flip=True,shear_range=0.1)

train = train_generator.flow_from_directory("data/train",target_size=(img_height,img_width),batch_size=32,class_mode='categorical')

"""
plt.figure(figsize=(10,10))
for image in enumerate(train_generator):
    plt.subplot(6,6)
    plt.imshow(image)
"""

img=load_img("data/train/A/a01.bmp")
x=img_to_array(img)
x=x.reshape((1,)+x.shape)
print(x.shape)
i=0
for batch in train_generator.flow(x,batch_size=1, save_to_dir='preview', save_prefix='preview', save_format='bmp'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely


# In[5]:


val_generator = ImageDataGenerator()
validation = val_generator.flow_from_directory('data/validation',target_size=(img_height,img_width),batch_size=32,class_mode="categorical")


# In[6]:


model = Sequential([
    BatchNormalization(axis=1, input_shape=(img_width,img_height,3)),
    Convolution2D(32, (3,3), activation='relu'),
    BatchNormalization(axis=1),
    MaxPooling2D(),
    Convolution2D(64, (3,3), activation='relu'),
    BatchNormalization(axis=1),
    MaxPooling2D(),
    Convolution2D(128, (3,3), activation='relu'),
    BatchNormalization(axis=1),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(5, activation='sigmoid')
])
model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Now we train the model

# In[7]:


model.fit_generator(train,steps_per_epoch=train.samples//32,epochs=60,validation_data=validation,validation_steps=validation.samples)


# In[8]:


model.save_weights('augmented_result.h5')


# In[9]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# In[10]:


totalepochs=100
epoch=0
train = train_generator.flow_from_directory("Handgesturedata/",target_size=(img_height,img_width),batch_size=500,class_mode='sparse')
for i,j in enumerate(train):
    X = j[0]
    y= j[1]
    break;
counter=0
for i,j in enumerate(train):
    if counter > 10:
        break
    X = np.append(X, j[0],axis=0)
    y = np.append(y, j[1],axis=0)
    counter+=1

#use Sklearn algorithms here
X = X.reshape((len(X),-1))
print("Shape of data after flattening the image: ", X.shape)
print("Shape of the target value: ", y)

#Polynomial features expansion to degree 2
#poly = PolynomialFeatures(degree=2,interaction_only=True)
#X = poly.fit_transform(X)

#radial basis function kernel over the transformed data
#rbf_feature = RBFSampler(gamma=1, random_state=1)
#X = rbf_feature.fit_transform(X)


#feature selection using Lasso
lsvc = svm.LinearSVC(C=0.01).fit(X,y)
skmodel = SelectFromModel(lsvc, prefit=True)
#ard = ARDRegression()
#skmodel = SelectFromModel(ard, prefit=True)
X = skmodel.transform(X)

print("Shape of data after feature selection: ", X.shape)
#we intiate the classifier objects here
#decreasing the number of principal components to be lesser than the number of training examples.
pca = PCA(n_components=1000)
X=pca.fit_transform(X)

print("Shape of data after applying PCA: ", X.shape)


# In[11]:


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
kf = KFold(n_splits=3)
kf.get_n_splits(X)
print(kf)
finalaccuracy=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    xgmodel = XGBClassifier()
    xgmodel.fit(X_train, y_train)
    y_pred = xgmodel.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy*100.0)
    finalaccuracy.append(accuracy*100.0)
print("Accuracy: ", sum(finalaccuracy)/float(len(finalaccuracy)))


# In[12]:


from sklearn.ensemble import AdaBoostClassifier
finalaccuracy=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y0_train, y_test = y[train_index], y[test_index]
    svmadamodel = AdaBoostClassifier(base_estimator=svm.SVC(probability=True, kernel="poly", degree=5), n_estimators=50)
    svmadamodel.fit(X_train, y_train)
    y_pred = svmadamodel.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy*100.0)
    finalaccuracy.append(accuracy*100.0)
print("Accuracy: ", sum(finalaccuracy)/float(len(finalaccuracy)))


# In[13]:


from sklearn.ensemble import RandomForestClassifier
finalaccuracy=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rfcmodel =  RandomForestClassifier(n_estimators=100)
    rfcmodel.fit(X_train, y_train)
    y_pred = rfcmodel.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy*100.0)
    finalaccuracy.append(accuracy*100.0)
print("Accuracy: ", sum(finalaccuracy)/float(len(finalaccuracy)))


# Here, we load the test pictures and calculate the accuracy.

# In[14]:


path = ("test/")
img_width, img_height = 100, 100
batchsize=32
for subfolder in os.listdir(path):
    print(subfolder)
    for item in os.listdir(path+subfolder):
        #print(item)
        if item.endswith(".bmp"):
            im = Image.open(path+subfolder+"/"+item)
            print(path+subfolder+"/"+item)
            f, e = os.path.split(path+subfolder+"/"+item)
            imResize = im.resize((100,100), Image.ANTIALIAS)
            im.close()
            imResize.save(path+subfolder+"/"+item,"JPEG")


# In[15]:


epoch=0
test_generator = ImageDataGenerator()
test = test_generator.flow_from_directory("test/",target_size=(img_height,img_width),batch_size=10,class_mode='categorical')
for i,j in enumerate(test):
    X_test = j[0]
    y_test = j[1]
    print(X_test.shape)
    print(y_test.shape)
    break;


# Now we test the test data on the Neural Network.

# In[16]:


print(model.metrics_names)
model.test_on_batch(X_test, y_test)


# In[17]:


epoch=0
test_generator = ImageDataGenerator()
test = test_generator.flow_from_directory("test/",target_size=(img_height,img_width),batch_size=10,class_mode='sparse')
for i,j in enumerate(test):
    X_test = j[0]
    y_test = j[1]
    break;

#use Sklearn algorithms here
X_test2 = X_test.reshape((len(X_test),-1))

#ard = ARDRegression()
#skmodel = SelectFromModel(ard, prefit=True)
X_test2 = skmodel.transform(X_test2)
#we intiate the classifier objects here
#decreasing the number of principal components to be lesser than the number of training examples.
#pca = PCA(n_components=1000)
X_test2=pca.transform(X_test2)


# Now, we test the data on non Neural Network based algorithms which were trained above.

# In[18]:


#We test XGboost classifier on the test set
y_pred = xgmodel.predict(X_test2)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print(accuracy*100.0)

#We test SVM with adaboost on the test set
y_pred = svmadamodel.predict(X_test2)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print(accuracy*100.0)

#We train random forest classifier on the test set
y_pred = rfcmodel.predict(X_test2)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print(accuracy*100.0)

