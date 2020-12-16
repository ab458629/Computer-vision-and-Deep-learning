'''
HW02_05.py
航太所碩一 P46091204 蔡承穎  Copyright (C) 2020

利用Pytorch torchvision 中的 ResNet50 達成貓狗分類 (資料儲存方式如下)
./data
    ./train
        ./cat
        ./dog
    ./validation
        ./cat
        ./dog
'''
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import time


NAME = "Cat-vs-Dos-ResNet50-{}".format(int(time.time()))  # 避免overwritting

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=1)

DATASET_PATH = 'data'
IMAGE_SIZE = (224, 224)  # 官方也是用這個尺寸
NUM_CLASSES = 2
BATCH_SIZE = 16  # RTX2060 也不能太大，只有6GB而已
FREEZE_LAYERS = 2
NUM_EPOCHS = 5

'''
圖像預處理 (Data Agumentation => ImageDataGenerator)
    (1) width_shift_range 和 height_shift_range 設定影象平移, 移出去的會用邊緣的畫素來補充(nearest)，盡量不要太大，否則物體飄出去
    (2) rotation_range 設定影象翻轉, 在[0, 指定角度]內進行隨機旋轉
    (3) shear_range 就是shear strain的概念
    (4) brightness_range 設定影象亮度
    (5) zoom_range 設定影象縮放, 0~1是放大，大於1是縮小
    (6) channel_shift_range 意思是整張圖都呈現某種顏色，變色片的概念
    (7) horizontal_flip 隨機對某幾張照片水平翻轉，貓狗不要用垂直，沒有動物是倒過來的
'''

train_datagen = ImageDataGenerator(rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',  # 2D one-hot
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()

valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/validation',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)


for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

# 利用tensorflow裡的ResNet開始建置模型, 不包含最後全連
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# include_top=False 自己補上output layer
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

# 因為微調，所以我們得先凍結幾層
net_final = Model(inputs=net.input, outputs=x)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True


net_final.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# print(net_final.summary())

net_final.fit_generator(train_batches,
                        steps_per_epoch=train_batches.samples // BATCH_SIZE,
                        validation_data=valid_batches,
                        validation_steps=valid_batches.samples // BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        callbacks=[tensorboard])

# 儲存參數 HDF5 file
net_final.save('HW02_05_ResNet50.h5')
