import pandas as pd
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from TCA import TCA
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('D:/Code/java/synapse_data/test_composite_data.csv')
test_data = pd.read_csv('D:/Code/java/lucene_data/test_composite_data.csv')
#读取训练和测试的X和Y
x_train = train_data.iloc[:, 1:41]
x_test = test_data.iloc[:, 1:41]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

#SMOTE算法部分
#检索数据集中缺陷标签为1的数据，标签为y_defects,词向量为X_defects
#使用SMOTE算法对缺陷标签为1的样本进行合成,defect_samples_resampled, _ 代表经过SMOTE算法平衡后的缺陷样本数据集和标签
smote = SMOTE(k_neighbors=5, random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_resampled, y_train_resampled, random_state=11)
#修改y的类型
y_train = K.cast_to_floatx(y_train)
y_test = K.cast_to_floatx(y_test)
y_valid = K.cast_to_floatx(y_valid)

x_train_tca, x_test_tca = TCA().fit(x_train, x_test)
#print(x_train_tca.shape, x_test_tca.shape, type(x_train_tca), type(x_test_tca))
#数据标准化
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train_tca)
x_test_scaler = scaler.fit_transform(x_test_tca)
x_valid_scaler = scaler.fit_transform(x_valid)
#数据缩放至[0,1]
max_abs_scaler = MaxAbsScaler()
x_train = max_abs_scaler.fit_transform(x_train_scaler)
x_test = max_abs_scaler.fit_transform(x_test_scaler)
x_valid = max_abs_scaler.fit_transform(x_valid_scaler)
#将二维数据转换为三维数据(np数据）
x_train = x_train[:, np.newaxis, :]
x_test = x_test[:, np.newaxis, :]
x_valid = x_valid[:, np.newaxis, :]


features = 36#每个时间步的特征数量
#创建模型
input_layers = tf.keras.layers.Input(shape=[1, features])
# 双向 LSTM 层
lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation='relu'))(input_layers)
droup_out_1 = tf.keras.layers.Dropout(0.2)(lstm_out)
attention_out = tf.keras.layers.Attention()([droup_out_1, droup_out_1])
concat_out = tf.keras.layers.Concatenate()([droup_out_1, attention_out])
droup_out_2 = tf.keras.layers.Dropout(0.2)(concat_out)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(droup_out_2)
model = tf.keras.models.Model(inputs=input_layers, outputs=output_layer)
# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, batch_size=1000)

pred_y_test = model.predict(x_test)
pred_y_test = np.round(pred_y_test)

precision = metrics.precision_score(y_test, pred_y_test)
recall = metrics.recall_score(y_test, pred_y_test)
f1 = metrics.f1_score(y_test, pred_y_test)
print(precision, recall, f1)





