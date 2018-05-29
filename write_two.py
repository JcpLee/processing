import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
j = -1
base_path = '/media/hlee/Document/demo/validation_age'
filename = 'validation.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)


path_o = os.listdir(base_path+'/')
# for i in path_o:#有多少类别
with tf.Session() as sess:
    s = 0
    j = j+1
    # print('%d'%j)
    p1 = path_o[0]
    p2 = path_o[1]
    lis1 = os.listdir(base_path + '/' + p1 + '/')
    lis2 = os.listdir(base_path + '/' + p2 + '/')

    for k in range(len(lis1)):
        path_1 = base_path+'/'+p1+'/'+lis1[k]
        path_2 = base_path+'/'+p2+'/'+lis2[k]

        s = s+1
        img1 = Image.open(path_1)
        img1 = img1.resize((227, 227),Image.ANTIALIAS)
        img1 = img1.convert('L')
        h, w = img1.size
        # print(img)
        img1 = np.asarray(img1)

        img2 = Image.open(path_2)
        img2 = img2.resize((227, 227), Image.ANTIALIAS)
        img2 = img2.convert('L')
        # # img = test_tf.image.resize_images(img, 1000, 800, method=0)
        b = np.zeros((img1.shape[0], img1.shape[1]), dtype=img1.dtype)
        g = np.zeros((img1.shape[0], img1.shape[1]), dtype=img1.dtype)

        #在第三维上累加
        b[:, :] = img1
        g[:, :] = img2
        img = np.dstack([b, g])


        # print(h,w)
        #
        # # img = test_tf.reshape(img,[w,h])
        # # # print(sess.run(img))
        # # img = sess.run(img).tobytes()
        # # print(sess.run(img).shape)
        img = img.tobytes()
        example = tf.train.Example(features = tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            'channels':tf.train.Feature(int64_list = tf.train.Int64List(value=[2]))
        }
        ))
        writer.write(example.SerializeToString())
    print(s)
    print('\n')
writer.close()
        # img1 = Image.open(path)

        # img1 = img1.convert('RGB')
        # if(j==1):
        #
        #     print(img1.size,img1.format,img1.mode)
        #
        #     # plt.imshow(sess.run(img).reshape(arr), cmap='gray')
        #
        #     plt.imshow(img1)
        #     plt.show()

        # print(path)