import os
import itertools
from PIL import Image



import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import confusion_matrix, classification_report



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

import os
os.environ['PYTHONUTF8'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


data_dir = './chest_xray'
filepaths = []
labels = []

folds = os.listdir(data_dir)

print(folds)




folder_dictionary = {}




for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    if os.path.isdir(foldpath):  # Check if foldpath is a directory
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            if os.path.isdir(fpath):  # Check if fpath is a directory
                image_path_list = os.listdir(fpath)
                if fold in folder_dictionary.keys():
                    folder_dictionary[fold] += len(image_path_list)
                else:
                    folder_dictionary[fold] = len(image_path_list)
                for img in image_path_list:
                    img_filepaths = os.path.join(fpath, img)
                    filepaths.append(img_filepaths)
                    labels.append(file)




folder_dictionary




filepaths[:5]




labels[:5]




Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis= 1)

for path in df['filepaths']:
    if not os.path.exists(path):
        print(f"Invalid path: {path}")
def safe_load_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()  # Verify image integrity
        return True
    except:
        return False

df['valid'] = df['filepaths'].apply(safe_load_image)
df = df[df['valid']]



img_normal = Image.open('chest_xray\\test\\NORMAL\\IM-0001-0001.jpeg')




plt.imshow(img_normal)




img_pnuemonia = Image.open('chest_xray/train/PNEUMONIA/person3_bacteria_10.jpeg')




plt.imshow(img_pnuemonia)




df.head()




test_df = df.iloc[:folder_dictionary['test'],:]




train_df = df.iloc[folder_dictionary['test']:(folder_dictionary['test']+folder_dictionary['train']),:]




validation_df = df.iloc[(folder_dictionary['test']+folder_dictionary['train']):(folder_dictionary['test']+folder_dictionary['train'] +folder_dictionary['val'] ),:]




train_df.iloc[2000]




batch_size = 32
img_size = (224, 224)

tr_gen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,)
ts_gen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,)
val_gen= ImageDataGenerator(rescale=1./255,horizontal_flip=True,)

train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', 
                                       target_size= img_size, class_mode= 'categorical',
                                       color_mode= 'rgb', shuffle= True, batch_size= batch_size)

valid_gen = val_gen.flow_from_dataframe( validation_df, x_col= 'filepaths', y_col= 'labels',
                                        target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)

test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels',
                                      target_size= img_size, class_mode= 'categorical',
                                      color_mode= 'rgb', shuffle= False, batch_size= batch_size)



def create_model(img_size):
    
    model = Sequential([
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape= (img_size[0],img_size[1],3)),
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
  
        Dense(256,activation = "relu"),
        Dense(64,activation = "relu"),
        Dense(1, activation = "sigmoid"),
        Dense(2, activation='softmax')
    ])
    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model




strategy = tf.distribute.MirroredStrategy()




with strategy.scope():
    model = create_model(img_size)
    model.compile(Adamax(learning_rate= 0.001), loss= 'binary_crossentropy', metrics= ['accuracy'])

model.summary()

# Define the number of epochs and the steps per epoch
epochs = 1 #use 10 for optimal solutions , our CPU can handle only 1 to 3 , can use GPU for 10
steps_per_epoch = train_gen.samples // batch_size
validation_steps = valid_gen.samples // batch_size

# Train the model
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Predicting on the test set
predictions = model.predict(test_gen, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
# Save the trained model to a file
model.save('pneumonia_detection_model.h5')

# Get the true classes
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())   

# Generate a classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plot Confusion Matrix
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=0  # Use verbose=1 or 2 if you want progress logs
)
plt.close()


# # Example of using the trained model to predict an individual image
# def predict_image(image_path, model, img_size):
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize(img_size)
#     img_array = np.array(img) / 255.0  # Normalize the image
#     img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit model input
    
#     prediction = model.predict(img_array)
#     predicted_class = class_labels[np.argmax(prediction)]
    
#     plt.imshow(img)
#     plt.title(f"Predicted: {predicted_class}")
#     plt.show()
    
#     return predicted_class

# # Example prediction
# image_path = 'chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg'
# predicted_class = predict_image(image_path, model, img_size)
# print(f'The predicted class for the input image is: {predicted_class}')

 