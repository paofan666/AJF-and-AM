import pandas as pd
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn import metrics


train_data = pd.read_csv('D:/Code/java/synapse_data/train_composite_data.csv')
test_data = pd.read_csv('D:/Code/java/synapse_data/test_composite_data.csv')
#Read the X and Y of training and testing
x_train = train_data.iloc[:, 1:41]
x_test = test_data.iloc[:, 1:41]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

#SMOTE algorithm section
#The data in the dataset is retrieved with a defect label of 1, labeled y_defects, and a word vector of X_defects
#The sample with defect label 1 is synthesized using the SMOTE algorithm, defect_samples_resampled, _ represents the defect sample dataset and labels balanced by the SMOTE algorithm
smote = SMOTE(k_neighbors=5, random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

x_train, x_valid, y_train, y_valid = train_test_split(x_train_resampled, y_train_resampled, random_state=11)
print(x_train.shape,x_valid.shape)
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

features = 40#The number of features per time step
#Create a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(40, input_shape=(1, features)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 二分类问题，使用sigmoid激活函数
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=5, batch_size=1000)

pred_y_test = model.predict(x_test)
pred_y_test = np.round(pred_y_test)

precision = metrics.precision_score(y_test, pred_y_test)
recall = metrics.recall_score(y_test, pred_y_test)
f1 = metrics.f1_score(y_test, pred_y_test)
print(precision, recall, f1)





