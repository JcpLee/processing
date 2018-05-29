import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
j = -1
base_path = '/media/hlee/Document/demo/validation_age'
filename = 'validation.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)

sess = tf.Session()
for i in os.listdir(base_path+'/'):#有多少类别

    s = 0
    print(i)
    j = j+1
    print('%d'%j)
    for k in os.listdir(base_path+'/'+i+'/'):
        path = base_path+'/'+i+'/'+k
        s = s+1
        # print(path)

        img = Image.open(path)
        img = img.resize((227, 227),Image.ANTIALIAS)
        img = img.convert('RGB')
        # print(img)
        # # img = test_tf.image.resize_images(img, 1000, 800, method=0)
        #
        h,w = img.size
        # print(s)
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
            'channels':tf.train.Feature(int64_list = tf.train.Int64List(value=[3]))
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