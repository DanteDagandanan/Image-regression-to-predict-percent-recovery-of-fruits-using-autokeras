#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:36:29 2023

This code loads the images of jackfruit and their corresponding mass information 
from a directory and a CSV file respectively. It then normalizes the mass values 
and splits the data into training and validation sets using the train_test_split()
 function from the sklearn.model_selection library.
It then Initializes an Image Regressor from the AutoKeras library and fit the model
 with the dataset, labels and time_limit of 126060 seconds.
It then evaluate the model on test data and prints the test loss and test accuracy.

Please note that this is just an example and you may need to adjust the code
 depending on how your data is organized and how you want to set the parameters
 for the model. Also, You will have to install sklearn package if you don't have 
it already.

@author: engrimmanuel

pip install tensorflow==2.9.0
pip install matplotlib==3.6.2
pip install autokeras==1.0.19
pip install os-sys==2.1.2
pip install opencv-python==4.6.0.66
pip install Pillow==9.3.0
pip install pandas==1.5.1


"""



import keras
import tensorflow as tf

from tensorflow.keras.utils import normalize
from autokeras import ImageRegressor

import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd




#loading the images save at csv file

df=pd.read_csv("C:/Users/PRO/Desktop/dante kadlfjasd/JACKFRUITS PROJECT FILES/Jackfruit Technology Project (2021-2023)/Data_Jackfruit Tech Project_21-23/development/jackfruit images/image classifcation tasks/code/no-cracks-jackfruits-percent-PerRecover-V-M.csv")

df1 = df.iloc[0:3577] 

path=df1["path image-1-to-83-"].values

mass= df1["mass (kg)"].values

volume= df1["volume (L)"].values

percent_recovery = df1["% recovery"].values

pulp_mass = df1["pulp mass (kg)"]


##########################################################

df2 = df.iloc[3577:4088]

path2=df2["path image-1-to-83-"].values

mass2= df2["mass (kg)"].values

volume2= df2["volume (L)"].values

percent_recovery2 = df2["% recovery"].values

pulp_mass2 = df2["pulp mass (kg)"]

######################################################

df3 = df.iloc[4088:4527]

path3=df3["path image-1-to-83-"].values

mass3= df3["mass (kg)"].values

volume3= df3["volume (L)"].values

percent_recovery3 = df3["% recovery"].values

pulp_mass3 = df3["pulp mass (kg)"]

######################################################
#loading the images

images = []

#image = cv2.imread("/home/engrimmanuel/Desktop/jackfruit task image regression task /no cracks RGB images/1Sample_RBG_images_0degree.png")

# if images is not None:
#     print('mage is loaded correctly')
#     # perform image processing
# else:
#     # image is not loaded correctly
#     print("Image not found or can't be read")

# #cv2.imshow("dante",image)


# Iterate over the rows of the DataFrame
for image_path in path:
   
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    #image = image.resize((256,256))
    image = image.resize((240,320))
    
    #image = Image.open(image_path)
    #image_array = np.array(image)
        
    # Append the image to the list
    #images.append(image)
    images.append(np.array(image))

images1_scaled = np.array(images)/255.0

######################################################################

images2 = []

# Iterate over the rows of the DataFrame
for image_path2 in path2:
   
    # Load the image using OpenCV
    image2 = cv2.imread(image_path2)
    image2 = Image.fromarray(image2, 'RGB')
    #image = image.resize((256,256))
    image2 = image2.resize((240,320))
    images2.append(np.array(image2))

image2_scaled = np.array(images2)/255.0

######################################################################
images3 = []
for image_path3 in path3:
   
    # Load the image using OpenCV
    image3 = cv2.imread(image_path3)
    image3 = Image.fromarray(image3, 'RGB')
    #image = image.resize((256,256))
    image3 = image3.resize((240,320))
    
    #image = Image.open(image_path)
    #image_array = np.array(image)
        
    # Append the image to the list
    #images.append(image)
    images3.append(np.array(image))

images3_scaled = np.array(images3)/255.0

######################################################################


# # Split data into training and validation sets
# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(images1_scaled,pulp_mass,test_size=0.30)

# X_val, X_test, y_val, y_test = train_test_split(X_val,  y_val, test_size=0.50)

# # Normalize the data
# X_train = np.array(X_train)/255.0
# X_val = np.array(X_val)/255.0
# X_test = np.array(X_test)/255.0
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# y_val = np.array(y_val)

def r_squared(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

# Initialize the image regressor and fit the data
#see the documantation here: https://autokeras.com/image_regressor/
regressor = ImageRegressor(directory="C:/Users/PRO/Desktop/", max_trials=5,overwrite=True)

regressor.fit(x=images1_scaled, y=pulp_mass,epochs=50,batch_size=32,validation_data=(image2_scaled, pulp_mass2))

#model_regressor= tf.keras.models.load_model('C:/Users/PRO/Desktop/dante kadlfjasd/JACKFRUITS PROJECT FILES/Jackfruit Technology Project (2021-2023)/Data_Jackfruit Tech Project_21-23/development/jackfruit images/pulp preduction R2 0.9947 MSE 0.0176 MAE 0.094 image.h5')

# Evaluate the model on the test data
test_loss, test_acc = regressor.evaluate(x=image2_scaled, y=pulp_mass2)
#test_loss, test_acc = model_regressor.evaluate(x=X_test, y=y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # Make predictions calibration
# y_pred_cal = regressor.predict(X_val)
# #y_pred_cal = model_regressor.predict(X_val)

# # Calculate MAE, MSE, and R^2
# mae_val = mean_absolute_error(y_val, y_pred_cal)
# mse_val = mean_squared_error(y_val, y_pred_cal)
# r2_val = r2_score(y_val, y_pred_cal)

# # Print the results
# print("Mean Absolute Error Calibration:", mae_val)
# print("Mean Squared Error Calibration:", mse_val)
# print("R-Squared Calibration:", r2_val)

# ########################################################

# # Make predictions
# y_pred_pre = regressor.predict(X_test)
# #y_pred_pre = model_regressor.predict(X_test)

# # Calculate MAE, MSE, and R^2
# mae_pred = mean_absolute_error(y_test, y_pred_pre)
# mse_pred = mean_squared_error(y_test, y_pred_pre)
# r2_pred = r2_score(y_test, y_pred_pre)

# # Print the results
# print("Mean Absolute Error prediction:", mae_pred)
# print("Mean Squared Error prediction:", mse_pred)
# print("R-Squared prediction:", r2_pred)

################################################################

# Make predictions new datasets
y_pred_new = regressor.predict(images3_scaled)
#y_pred_new = model_regressor.predict(image_scaled)

# Calculate MAE, MSE, and R^2
mae_new = mean_absolute_error(np.array(pulp_mass3), y_pred_new)
mse_new = mean_squared_error(np.array(pulp_mass3), y_pred_new)
r2_new = r2_score(np.array(pulp_mass3), y_pred_new)

# Print the results
print("Mean Absolute Error image from 84-88:", mae_new)
print("Mean Squared Error image from 84-88:", mse_new)
print("R-Squared image from 84-88:", r2_new)

################################################################

# Make predictions new datasets
y_pred_calb = regressor.predict(image2_scaled)
#y_pred_new = model_regressor.predict(image_scaled)

# Calculate MAE, MSE, and R^2
mae_calb = mean_absolute_error(np.array(pulp_mass2),y_pred_calb)
mse_calb = mean_squared_error(np.array(pulp_mass2), y_pred_calb)
r2_calb = r2_score(np.array(pulp_mass2), y_pred_calb)

# Print the results
print("Mean Absolute Error image from 84-88 calibration:", mae_calb)
print("Mean Squared Error image from 84-88 calibration:", mse_calb)
print("R-Squared image from 84-88 calibration:", r2_calb)

################################################################

model = regressor.export_model()
model.summary()

import visualkeras
from PIL import ImageFont

visualkeras.layered_view(model,legend=True) 

from tensorflow.keras.utils import plot_model
# Visualize the model
#plot_model(model, to_file='C:/Users/PRO/Desktop/percent recovery models/image regression percent  recovery of jackfruits r2-'+ str(round(r2,2))+ ' MSE '+ str(round(mae,2)) + ' MAE: ' + str(round(mae,2)) + ' resize from 640 x 480 to 240 x 320 model.png', show_shapes=True, show_layer_names=True)

#model.save('C:/Users/PRO/Desktop/dante kadlfjasd/JACKFRUITS PROJECT FILES\Jackfruit Technology Project (2021-2023)/Data_Jackfruit Tech Project_21-23/development/jackfruit images/R-Squared 0.9947 Mean Squared Error 0.0176 Mean Absolute Error 0.094 image.h5')
#model.save('C:/Users/PRO/Desktop/percent recovery models/image regression percent recovery of jackfruits r2-'+ str(round(r2,2))+ ' MSE '+ str(round(mae,2)) + ' MAE: ' + str(round(mae,2)) + ' resize from 640 x 480 to 240 x 320 model.h5')
