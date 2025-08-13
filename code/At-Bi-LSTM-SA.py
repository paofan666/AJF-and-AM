import pandas as pd
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn import metrics

train_data = pd.read_csv('D:/Code/java/log4j_data/test_composite_data.csv')
test_data = pd.read_csv('D:/Code/java/lucene_data/test_composite_data.csv')
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

#Converting 2D data to 3D data (np data)
x_train = x_train[:, np.newaxis, :]
x_test = x_test[:, np.newaxis, :]
x_valid = x_valid[:, np.newaxis, :]


features = 40 #Number of features for each time step j
#Create a model
input_layers = tf.keras.layers.Input(shape=(1, 40))
# Bidirectional LSTM layer
lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation='relu'))(input_layers)
droup_out_1 = tf.keras.layers.Dropout(0.2)(lstm_out)
attention_out = tf.keras.layers.Attention()([droup_out_1, droup_out_1])
concat_out = tf.keras.layers.Concatenate()([droup_out_1, attention_out])
droup_out_2 = tf.keras.layers.Dropout(0.2)(concat_out)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(droup_out_2)
model = tf.keras.models.Model(inputs=input_layers, outputs=output_layer)
#model.summary()
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=300, batch_size=32)


pred_y_test = model.predict(x_test)
pred_y_test = np.round(pred_y_test)

'''print(pred_y_test.shape)
print(y_test.shape)
print(type(pred_y_test))
print(type(y_test))'''

precision = metrics.precision_score(y_test, pred_y_test)
recall = metrics.recall_score(y_test, pred_y_test)
f1 = metrics.f1_score(y_test, pred_y_test)
print(precision, recall, f1)





