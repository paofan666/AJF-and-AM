import pandas as pd
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from TCA import TCA

train_data = pd.read_csv('D:/Code/java/poi_data/train_composite_data_ast.csv')
test_data = pd.read_csv('D:/Code/java/veloctiy_data/train_composite_data_ast.csv')
#Read the X and Y of training and testing
x_train = train_data.iloc[:, 2:42]
x_test = test_data.iloc[:, 2:42]
y_train = train_data.iloc[:, 1]
y_test = test_data.iloc[:, 1]

#SMOTE algorithm section
#The data in the dataset is retrieved with a defect label of 1, labeled y_defects, and a word vector of X_defects
#The sample with defect label 1 is synthesized using the SMOTE algorithm, defect_samples_resampled, _ represents the defect sample dataset and labels balanced by the SMOTE algorithm
smote = SMOTE(k_neighbors=5, random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

x_train, x_valid, y_train, y_valid = train_test_split(x_train_resampled, y_train_resampled, random_state=11)

x_train_tca, x_test_tca = TCA().fit(x_train, x_test)
#Modify the type of y
y_train = K.cast_to_floatx(y_train)
y_test = K.cast_to_floatx(y_test)
y_valid = K.cast_to_floatx(y_valid)
#Data standardization
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train_tca)
x_test_scaler = scaler.fit_transform(x_test_tca)
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

# Construct a convolutional neural network model with a self-attention layer
def build_cnn_with_attention(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 1, activation='relu')(inputs)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    # Add a self-attention layer
    attention = tf.keras.layers.Attention()([x, x])
    # Connect the output of the self-attention layer with the original output
    x = tf.keras.layers.Concatenate()([x, attention])
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # 二分类问题的输出层
    model = tf.keras.Model(inputs, outputs)
    return model

# Build a model
input_shape = (x_train.shape[1], x_train.shape[2])
model = build_cnn_with_attention(input_shape)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=300, batch_size=32)

pred_y_test = model.predict(x_test)
pred_y_test = np.round(pred_y_test)
accuracy = metrics.accuracy_score(y_test, pred_y_test)
precision = metrics.precision_score(y_test, pred_y_test)
recall = metrics.recall_score(y_test, pred_y_test)
f1 = metrics.f1_score(y_test, pred_y_test)
print(precision, recall, f1)

