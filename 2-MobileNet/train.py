import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from MobileNet import MobileNet

BATCH_SIZE = 8
EPOCH = 5
NUM_CLASS = 2

def load_img(img_path,size=(224,224)):
    # 打标签
    if tf.strings.regex_full_match(img_path,".*cats.*"):
        label = tf.constant([0,1],tf.int8)
    else:
        label = tf.constant([1,0],tf.int8)
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
# 展示数据
# plt.figure(figsize=(8,8))
# for i,(img,label) in enumerate(ds_train.unbatch().take(9)):
    # ax = plt.subplot(3,3,i+1)
    # ax.imshow(img.numpy())
    # ax.set_title("label = %d"%label)
    # ax.set_xticks([])
    # ax.set_yticks([]) 
# plt.show()

# 获得模型
model = VGG16(2)
# 注意要开启skip_mismatch和by_name
model.load_weights("model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",skip_mismatch=True,by_name=True)

# 编译模型
model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=["accuracy"]
    )

# 训练模型
history = model.fit(ds_train,epochs=EPOCH,validation_data=ds_test)

# 保存模型最好采用H5格式，我觉得简单
model.save_weights('log/tf_model_weights.h5')
