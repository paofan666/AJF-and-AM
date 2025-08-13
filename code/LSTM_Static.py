import pandas as pd
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn import metrics

with open('D:/Code/java/xerces_data/xerces-1.2.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
    for row in data:
        # Treat defects in the csv file as 0 and 1
        if row[23] == 'bug':
            pass
        elif row[23] == '0':
            pass
        else:
            row[23] = 1
            with open('D:/Code/java/xerces_data/train_data_static.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)

with open('D:/Code/java/xerces_data/xerces-1.3.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
    for row in data:
        #print(row[2])
        # Treat defects in the csv file as 0 and 1
        if row[23] == 'bug':
            pass
        elif row[23] == '0':
            pass
        else:
            row[23] = 1
            with open('D:/Code/java/xerces_data/test_data_static.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)


train_data = pd.read_csv('D:/Code/java/xerces_data/train_data_static.csv')
test_data = pd.read_csv('D:/Code/java/xerces_data/test_data_static.csv')
#Read the X and Y of training and testing
x_train = train_data.iloc[:, 3:23]
x_test = test_data.iloc[:, 3:23]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

#SMOTE algorithm section
#The data in the dataset is retrieved with a defect label of 1, labeled y_defects, and a word vector of X_defects
#The sample with defect label 1 is synthesized using the SMOTE algorithm, defect_samples_resampled, _ represents the defect sample dataset and labels balanced by the SMOTE algorithm
smote = SMOTE(k_neighbors=5, random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
print(np.shape(x_train_resampled))

x_train, x_valid, y_train, y_valid = train_test_split(x_train_resampled, y_train_resampled, random_state=11)
#Modify the type of y
y_train = K.cast_to_floatx(y_train)
y_test = K.cast_to_floatx(y_test)
y_valid = K.cast_to_floatx(y_valid)
#Data standardization
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.fit_transform(x_test)
x_valid_scaler = scaler.fit_transform(x_valid)
#Data scaling to [0,1]
max_abs_scaler = MaxAbsScaler()
x_train = max_abs_scaler.fit_transform(x_train_scaler)
x_test = max_abs_scaler.fit_transform(x_test_scaler)
x_valid = max_abs_scaler.fit_transform(x_valid_scaler)

#Converting 2D data to 3D data (NP data)
x_train = x_train[:, np.newaxis, :]
x_test = x_test[:, np.newaxis, :]
x_valid = x_valid[:, np.newaxis, :]


#print(type(x_train_resampled))#<class 'pandas.core.frame.DataFrame'>


features = 20#The number of features per time step
#Create a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(100, input_shape=(1, features), return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dropout(0.2))
optimizer = tf.keras.optimizers.Adam(0.001)
# Add an output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 二分类问题，使用sigmoid激活函数
# Compile the model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=150, batch_size=128)

pred_y_test = model.predict(x_test)
pred_y_test = np.round(pred_y_test)

accuracy = metrics.accuracy_score(y_test, pred_y_test)
precision = metrics.precision_score(y_test, pred_y_test)
recall = metrics.recall_score(y_test, pred_y_test)
f1 = metrics.f1_score(y_test, pred_y_test)
auc = metrics.roc_auc_score(y_test, pred_y_test)
print(accuracy, precision, recall, f1, auc)





