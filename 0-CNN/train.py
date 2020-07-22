import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import datasets
from cnn import CNN
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------可修改的参数 --------
BATCH_SIZE = 8
EPOCH = 1
# ---------可修改的参数 --------

CLASS = os.listdir('data/train')
NUM_CLASS = len(CLASS)
def load_img(img_path,size=(32,32)):
    # 给数据打标签
    label = -1
    for i in range(NUM_CLASS):
        if tf.strings.regex_full_match(img_path,CLASS[i]):
            label = i
    label = tf.one_hot(label,NUM_CLASS)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img) #注意此处为jpeg格式
    img = tf.image.resize(img,size)/255.0
    return(img,label)

ds_train = tf.data.Dataset.list_files("data/train/*/*.jpg") \
        .map(load_img,num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
        .prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.Dataset.list_files("data/test/*/*.jpg") \
        .map(load_img,num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
        .prefetch(tf.data.experimental.AUTOTUNE)

# 获得模型
model = CNN()
# 编译模型
model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=["accuracy"]
    )
# 训练中每个世代保存一次
cp_callback = ModelCheckpoint(
        'logs/ep{epoch:02d}-loss{loss:.2f}.h5',
        monitor='acc',save_weights_only=True, 
    )
# 训练模型
history = model.fit(ds_train,epochs=EPOCH,
        validation_data=ds_test,
        callbacks=[cp_callback],
    )
# 保存最终模型
model.save_weights('logs/last1.h5')
