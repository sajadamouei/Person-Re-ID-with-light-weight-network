import keras
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Activation, Dropout, BatchNormalization, MaxPool2D, InputLayer
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import cv2
import matplotlib.image as mpimg
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import efficientnet.keras as efn
from keras.preprocessing.image import ImageDataGenerator
import pickle
import datetime

#======================================================== Load Dataset
npz= np.load('drive/My Drive/thesis/nparray camera/cuhk01.npz')
cuhk=npz['cuhk']
label=npz['label']

print(len(cuhk),len(label))

#======================================================== Resize
cuhk01=[]
#img=cuhk[0]
#scale_percent = 160 # percent of original size
width = 80 #int(img.shape[1] * scale_percent / 100)
height = 160 #int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  


for i in range(len(cuhk)):
  # resize image
  cuhk01.append(cv2.resize(cuhk[i], dim, interpolation = cv2.INTER_AREA))

cuhk=[]
cuhk=cuhk01
#======================================================== Augmentation
# cuhk_mr is horizontal_flip (mirror) dataset
cuhk_mr=[]
lbl_mr=[]
for i in range(3484):
  cuhk_mr.append(np.fliplr(cuhk[i]))
lbl_mr=label[0:3484]

print(len(cuhk_mr),len(lbl_mr))

#=================
def augmented(cuhk1,label1,s,w,h,z):
  #======================================
  datagen = ImageDataGenerator(shear_range=s,width_shift_range=w,height_shift_range=h,zoom_range=z,horizontal_flip=True)
  #==========================================
  train_data=cuhk1
  train_labels=label1
  train_data = np.squeeze(np.array(train_data))
  train_labels = np.squeeze(np.array(train_labels))
  #=================================================
  augmented_data = []
  augmented_label=[]
  batch_size=1
  for i in range(1,3485,1):
    num_augmented = 0
    one_image=train_data[i-1:i]
    one_label=train_labels[i-1:i]
    for X_batch, y_batch in datagen.flow(one_image, one_label, batch_size=batch_size, shuffle=False):
      augmented_data.append(X_batch)
      augmented_label.append(y_batch)
      num_augmented += batch_size
      if num_augmented >= 1:
          break
  augmented_data = np.concatenate(augmented_data)
  augmented_label = np.concatenate(augmented_label)
  return augmented_data, augmented_label
  
#==================
# cuhk augmented dataset
aug_data1,aug_lbl1=augmented(cuhk,label,0.2,0.2,0.1,0.2)
print(len(aug_data1),len(aug_lbl1))
aug_data2,aug_lbl2=augmented(cuhk,label,0.1,0.1,0.1,0.1)
print(len(aug_data2),len(aug_lbl2))
#==================

#======================================================== Create Pairs
def pairs(cuhk1,label1):

  #Number of pairs per image
  pairs = 2

  #Let's create the new dataset to train on
  leftp_train = []
  leftp_id = []
  rightp_id = []
  rightp_train = []
  targetsp = []
  #Positive Paires
  for i in range(0,3484,4):
      leftp_train.append(cuhk1[i])
      leftp_id.append(int(label1[i][0:4].lstrip('0'))-1)
      rightp_train.append(cuhk1[i+2])
      rightp_id.append(int(label1[i+2][0:4].lstrip('0'))-1)
      targetsp.append(1)

      leftp_train.append(cuhk1[i])
      leftp_id.append(int(label1[i][0:4].lstrip('0'))-1)
      rightp_train.append(cuhk1[i+3])
      rightp_id.append(int(label1[i+3][0:4].lstrip('0'))-1)
      targetsp.append(1)

      leftp_train.append(cuhk1[i+1])
      leftp_id.append(int(label1[i+1][0:4].lstrip('0'))-1)
      rightp_train.append(cuhk1[i+2])
      rightp_id.append(int(label1[i+2][0:4].lstrip('0'))-1)
      targetsp.append(1)

      leftp_train.append(cuhk1[i+1])
      leftp_id.append(int(label1[i+1][0:4].lstrip('0'))-1)
      rightp_train.append(cuhk1[i+3])
      rightp_id.append(int(label1[i+3][0:4].lstrip('0'))-1)
      targetsp.append(1)
      #Negative Paire   
      for _ in range(pairs):
         for j in range(4):
              compare_to = i
              while compare_to == i or compare_to == i+1 or compare_to == i+2 or compare_to == i+3: #Make sure it's not comparing to itself
                  compare_to = random.randint(0,3483)
              leftp_train.append(cuhk1[i+j])
              leftp_id.append(int(label1[i+j][0:4].lstrip('0'))-1)
              rightp_train.append(cuhk1[compare_to])
              rightp_id.append(int(label1[compare_to][0:4].lstrip('0'))-1)
              targetsp.append(0)
  return leftp_train, leftp_id, rightp_train ,rightp_id ,targetsp   
  
#======================
left_train, left_id, right_train ,right_id ,targets = pairs(cuhk,label)

t1, t2, t3 ,t4 ,t5 = pairs(cuhk_mr,lbl_mr)
left_train+=t1; left_id+=t2; right_train+=t3; right_id+=t4; targets+=t5

t1, t2, t3 ,t4 ,t5 = pairs(aug_data1,aug_lbl1)
left_train+=t1; left_id+=t2; right_train+=t3; right_id+=t4; targets+=t5

t1, t2, t3 ,t4 ,t5 = pairs(aug_data2,aug_lbl2)
left_train+=t1[:8000]; left_id+=t2[:8000]; right_train+=t3[:8000]; right_id+=t4[:8000]; targets+=t5[:8000]
left_val=t1[8000:]; left_vid=t2[8000:]; right_val=t3[8000:]; right_vid=t4[8000:]; val_targets=t5[8000:]

print('Validation Data= ',len(left_val),len(left_vid),len(right_val),len(right_vid),len(val_targets))
print('Train Data= ',len(left_train),len(left_id),len(right_train),len(right_id),len(targets))
#=========================

#================================================= Create query and gallery
######################################################################### Query and Gallery
query=[]
query_id=[]
gallery=[]
gallery_id=[]

#Create Query
for i in range(3484,len(cuhk),4):
  one_zero = random.randint(0,1)
  query.append(cuhk[i+one_zero])
  query_id.append(label[i+one_zero])

#Create Gallery
for i in range(3484,len(cuhk),4):
  one_zero = random.randint(2,3)
  gallery.append(cuhk[i+one_zero])
  gallery_id.append(label[i+one_zero])

#Create Test Pair
left_test=[]
right_test=[]
test_targets=[]

for i in range(len(query)):
  for j in range(len(gallery)):
    left_test.append(query[i])
    right_test.append(gallery[j])
    if i!=j:
      test_targets.append(0)
    else:
      test_targets.append(1)


left_test = np.array(left_test)
right_test = np.array(right_test)
test_targets= np.array(test_targets)
print('left_test shape: ',left_test.shape)
print('right_test shape: ',right_test.shape)
print('test_targets shape: ',test_targets.shape)

########################################################################### Evaluate
left_ev=[];right_ev=[];target_ev=[]

for i in range(len(query)):
  for j in range(len(gallery)):
    if i==j:
      compare_to = j
      left_ev.append(query[i])
      right_ev.append(gallery[j])
      target_ev.append(1)
      while compare_to == j: #Make sure it's not comparing to itself
        compare_to = random.randint(0,99)
      left_ev.append(query[i])
      right_ev.append(gallery[compare_to])
      target_ev.append(0)  

left_ev = np.array(left_ev)
right_ev = np.array(right_ev)
target_ev= np.array(target_ev)
print('left_ev shape: ',left_ev.shape)
print('right_ev shape: ',right_ev.shape)
print('target_ev shape: ',target_ev.shape)      


#========================================================== 
#==================================== Train np.array
left_train = np.array(left_train)
#left_id=[d.lstrip('0') for d in left_id]
left_id = np.array(left_id)
right_train = np.array(right_train)
#right_id=[d.lstrip('0') for d in right_id]
right_id = np.array(right_id)
targets = np.array(targets)
print('left_train shape: ',left_train.shape)
print('left_id shape: ',left_id.shape)
print('right_train shape: ',right_train.shape)
print('right_id shape: ',right_id.shape)
print('targets shape: ',targets.shape)
#==================================== Train To_categorical 
left_id = to_categorical(left_id)
right_id = to_categorical(right_id)
print('left_id to categorical: ',left_id.shape)
print('right_id to categorical: ',right_id.shape)

##################################################################################################################
#===================================== Validation np.array

left_val = np.array(left_val)
#left_id=[d.lstrip('0') for d in left_id]
left_vid = np.array(left_vid)
right_val = np.array(right_val)
#right_id=[d.lstrip('0') for d in right_id]
right_vid = np.array(right_vid)
val_targets = np.array(val_targets)
print('left_val shape: ',left_val.shape)
print('left_vid shape: ',left_vid.shape)
print('right_val shape: ',right_val.shape)
print('right_vid shape: ',right_vid.shape)
print('val_targets shape: ',val_targets.shape)
#===================================== Validation To_Categorical
left_vid = to_categorical(left_vid)
right_vid = to_categorical(right_vid)
print('left_id to categorical: ',left_vid.shape)
print('right_id to categorical: ',right_vid.shape)

#################################################################################################################
# Create fake_target class_l and class_r for evaluate model
fake_ltarget = left_id[0:10000][:]
fake_rtarget = right_id[0:10000][:]
print('fake target: ',fake_ltarget.shape)
print('fake target: ',fake_rtarget.shape)