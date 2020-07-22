import tensorflow as tf
from cnn import CNN

model = CNN()
model.load_weights('logs/last1.h5')

print("[Warning] 请把照片放在img文件夹下!")
while True:
    name = input("请输入照片名字:")
    try:
        img = tf.io.read_file("img/"+name)
        img = tf.image.decode_jpeg(img) #注意此处为jpeg格式
        img = tf.image.resize(img,(32,32))/255.0
        img = tf.expand_dims(img,axis=0)
        ans = model.predict(img)
        print("预测结果为:%d概率为%.3f"%(ans.argmax(),max(ans[0])))
    except:
        print("输入名称有误!")
