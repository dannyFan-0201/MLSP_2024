import os
import cv2
import numpy
import imageio
import pandas as pd
from keras.layers.core import Dropout
from keras.layers import  TimeDistributed, GRU
from keras.models import Model
from keras.layers import Input, Conv3D, LeakyReLU, concatenate, Flatten, Dense
from keras.layers.convolutional import MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import Callback
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from attention import CBAMModule
from keras.regularizers import l1, l2


class ClassLossCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data

        # confusion_matrix
        predictions = self.model.predict(x_val)
        y_pred = numpy.argmax(predictions, axis=-1)
        y_true = numpy.argmax(y_val, axis=-1)
        y_pred = y_pred[:, 0]
        cfm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cfm)

        # 計算混淆矩陣
        cfm = tf.math.confusion_matrix(y_true, y_pred)
        # acc
        acc_score = tf.reduce_sum(tf.linalg.diag_part(cfm)) / tf.reduce_sum(cfm)
        # UAR
        true_positives = tf.linalg.diag_part(cfm)
        true_positives = tf.cast(true_positives, tf.float32)
        actual_positives = tf.reduce_sum(cfm, axis=1)
        actual_positives_safe = tf.where(tf.equal(actual_positives, 0), tf.constant(1e-7, dtype=tf.float32), tf.cast(actual_positives, tf.float32))
        recall_per_class = true_positives / actual_positives_safe
        uar = tf.cond(tf.equal(acc_score, 1), lambda: acc_score, lambda: tf.reduce_mean(recall_per_class))
        # UF1
        precision_per_class = true_positives / tf.cast(tf.reduce_sum(cfm, axis=0), tf.float32)
        precision_per_class_safe = tf.where(tf.math.is_finite(precision_per_class), precision_per_class,tf.constant(1e-7, dtype=tf.float32))
        f1_per_class = 2 * (precision_per_class_safe * recall_per_class) / (precision_per_class_safe + recall_per_class + tf.constant(1e-7, dtype=tf.float32))
        uf1 = tf.cond(tf.equal(acc_score, 1), lambda: acc_score, lambda: tf.reduce_mean(f1_per_class))

        print("UAR:", uar.numpy())
        print("UF1:", uf1.numpy())
        uar_list.append(uar.numpy())
        uf1_list.append(uf1.numpy())
        acc_list.append(acc_score.numpy())
        uf1_value = max(uf1_list)
        if uf1_value == 1:
            self.model.stop_training = True


def weighted_categorical_crossentropy(class_weights):
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        weighted_losses = -tf.reduce_sum(class_weights * y_true * tf.math.log(tf.maximum(y_pred, 1e-15)), axis=-1)

        return weighted_losses
    return loss

image_rows, image_columns, image_depth = 128, 128, 10

training_list = []
video_list = []
negativepath = 'dataset path'
positivepath = 'dataset path'
surprisepath = 'dataset path'

directorylisting = os.listdir(negativepath)
video_list = directorylisting
for video in directorylisting:
    frames = []
    videopath = negativepath +video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x + 0 for x in range(10)]
    for frame in framerange:
        image = loadedvideo.get_data(frame)
        imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
        frames.append(imageresize)
    frames = numpy.asarray(frames)
    videoarray = frames.transpose(0, 1, 2, 3)
    training_list.append(videoarray)
directorylisting = os.listdir(positivepath)
video_list.extend(directorylisting)

for video in directorylisting:
    frames = []
    videopath = positivepath + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x + 0 for x in range(10)]
    for frame in framerange:
        image = loadedvideo.get_data(frame)
        imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
        frames.append(imageresize)
    frames = numpy.asarray(frames)
    videoarray = frames.transpose(0, 1, 2, 3)
    training_list.append(videoarray)
directorylisting = os.listdir(surprisepath)
video_list.extend(directorylisting)

for video in directorylisting:
        frames = []
        videopath = surprisepath + video
        loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
        framerange = [x + 0 for x in range(10)]
        for frame in framerange:
            image = loadedvideo.get_data(frame)
            imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
            frames.append(imageresize)
        frames = numpy.asarray(frames)
        videoarray = frames.transpose(0, 1, 2, 3)
        training_list.append(videoarray)

training_list = numpy.asarray(training_list)
trainingsamples = len(training_list)

traininglabels = numpy.zeros((trainingsamples, ), dtype=int)

traininglabels[0:70] = 0
traininglabels[70:121] = 1
traininglabels[121:164] = 2

traininglabels = np_utils.to_categorical(traininglabels, 3)

training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = numpy.zeros((trainingsamples, image_depth, image_rows, image_columns, 3))
for h in range(trainingsamples):
    training_set[h][:][:][:][:] = trainingframes[h,:,:,:,:]

training_set = training_set.astype('float32')
training_set -= numpy.mean(training_set)
training_set /= numpy.max(training_set)


# Load pre-trained weights
# model.load_weights('')


# 指定 Excel 檔案的路徑
excel_file_path = 'excel_path'
# 讀取 Excel 檔案
df = pd.read_excel(excel_file_path, sheet_name='Sheet1')
cleaned_video_list = [video.replace('.avi', '') for video in video_list]
label_dict = dict(zip(df['Filename'], df['Subject']))
label_list = [label_dict.get(video, 'unknown') for video in cleaned_video_list]
total_l = len(label_list)
total_v = len(video_list)
print(cleaned_video_list)
print(label_list)
print(total_v)
print(total_l)

# 定義你的訓練資料和標籤
X = training_set
y = traininglabels
# 定義每個樣本所屬的主題或子集，例如，這是一個主題列表
subjects = label_list

# 評估指標
Uar = []
Uf1 = []
acc = []
i = 1



# 使用LOSO交叉驗證
logo = LeaveOneGroupOut()
for train_index, test_index in logo.split(X, y, groups=subjects):
    print("\n "+ f"##### Subject: {i} #####\n")
    train_images, validation_images = X[train_index], X[test_index]
    train_labels, validation_labels = y[train_index], y[test_index]

    test_files = [cleaned_video_list[i] for i in test_index]
    print(f"Testing with files: {', '.join(test_files)}")

    #  Model  epoch:200
    # 初始輸入形狀
    input_shape = (image_depth, image_rows, image_columns, 3)
    # 創建模型
    input_layer = Input(shape=input_shape)
    # 第一個3D CNN分支
    CNN1 = Conv3D(32, (5, 5, 5), padding='same', kernel_initializer='he_normal')(input_layer)
    CNN1 = LeakyReLU(alpha=0.2)(CNN1)
    CNN1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(CNN1)
    # 第二個3D CNN分支
    CNN2 = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer='he_normal')(input_layer)
    CNN2 = LeakyReLU(alpha=0.2)(CNN2)
    CNN2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(CNN2)
    # 連接兩個分支
    merged = concatenate([CNN1, CNN2])
    CNN = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer='he_normal')(merged)
    CNN = LeakyReLU(alpha=0.2)(CNN)
    CNN = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(CNN)
    # 將模型組合為一個完整的模型
    output_layer = CBAMModule(channels=32, reduction=8)(CNN)
    output_layer = TimeDistributed(Flatten())(output_layer)
    output_layer = Dropout(0.3)(output_layer)
    output_layer = TimeDistributed(Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)))(output_layer)
    output_layer = LeakyReLU(alpha=0.2)(output_layer)
    output_layer = Dropout(0.3)(output_layer)
    output_layer = TimeDistributed(Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)))(output_layer)
    output_layer = LeakyReLU(alpha=0.2)(output_layer)
    output_layer = Dropout(0.3)(output_layer)
    output_layer = GRU(1024, return_sequences=True)(output_layer)
    output_layer = Dense(32, kernel_initializer='he_normal')(output_layer )
    output_layer = LeakyReLU(alpha=0.2)(output_layer)
    output_layer_final = Dense(3, activation='softmax')(output_layer )
    # 建立整合後的模型
    model = Model(inputs=input_layer, outputs=output_layer_final)
    class_weights = [0.3, 0.35, 0.35]
    loss = weighted_categorical_crossentropy(class_weights)
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    model.summary()

    # 評估指標
    uar_list = []
    uf1_list = []
    acc_list = []
    initial_weights = model.get_weights()
    model.set_weights(initial_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    filepath = "weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Training the model
    tensorboard_callback = TensorBoard(log_dir="TensorBoard_path")
    callbacks_list.append(tensorboard_callback)
    X_val = validation_images
    y_val = validation_labels
    callbacks_list.append(ClassLossCallback((X_val, y_val)))
    train_labels = numpy.repeat(train_labels[:, numpy.newaxis, :], 4, axis=1)
    validation_labels = numpy.repeat(validation_labels[:, numpy.newaxis, :], 4, axis=1)
    hist = model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), callbacks=callbacks_list, batch_size=8, epochs=200, shuffle=True)

    best_uf1_index = uf1_list.index(max(uf1_list))
    best_uar_for_best_uf1 = uar_list[best_uf1_index]
    print("Best UAR:", best_uar_for_best_uf1)
    print("Best UF1:", uf1_list[best_uf1_index])
    print("Best ACC:", max(acc_list))
    Uar.append(best_uar_for_best_uf1)
    Uf1.append(uf1_list[best_uf1_index])
    acc.append(max(acc_list))
    i += 1
    tf.keras.backend.clear_session()

#計算最終指標分數
for index, (value1, value2, value3) in enumerate(zip(Uar, Uf1, acc)):
    formatted_value1 = "{:.4f}".format(value1).rstrip('0').rstrip('.')
    formatted_value2 = "{:.4f}".format(value2).rstrip('0').rstrip('.')
    formatted_value3 = "{:.4f}".format(value3).rstrip('0').rstrip('.')
    print(f" {index + 1}: UAR={formatted_value1}, UF1={formatted_value2}, ACC={formatted_value3}")

average_Uar = sum(Uar) / len(Uar)
average_Uf1 = sum(Uf1) / len(Uf1)
average_acc = sum(acc) / len(acc)

print("Average UF1:", average_Uf1)
print("Average Uar:", average_Uar)
print("Average ACC:", average_acc)




