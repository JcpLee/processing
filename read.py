import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

path = 'validation.tfrecords'
files = tf.train.match_filenames_once(path)#获取所有符合正则表达式的文件,返回文件列表
filename_queue = tf.train.string_input_producer([path],shuffle=True)  # create a queue

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # return file_name and file

features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img': tf.FixedLenFeature([], tf.string),
                                       'width': tf.FixedLenFeature([], tf.int64),
                                       'height': tf.FixedLenFeature([], tf.int64),
                                       'channels':tf.FixedLenFeature([],tf.int64)
                                   })  # return image and label

# img = test_tf.image.convert_image_dtype(img, dtype=test_tf.float32)
# img = test_tf.reshape(img, [512, 80, 3])  # reshape image to 512*80*3
# img = test_tf.cast(img, test_tf.float32) * (1. / 255) - 0.5  # throw img tensor

label = tf.cast(features['label'], tf.int32)  # throw label tensor
# height = tf.cast(features['height'],tf.int32)
height = features['height']
width = tf.cast(features['width'],tf.int32)
channels = tf.cast(features['channels'],tf.int32)


img = tf.decode_raw(features['img'], tf.uint8)

img = tf.reshape(img,[227,227,2])





#
img_batch, label_batch = tf.train.shuffle_batch([img,label],
                                                batch_size=6,
                                                capacity=11,
                                                min_after_dequeue=10,
                                                num_threads=4)
#
# label_batch = tf.reshape(label_batch, [64,1])
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # sess.run(test_tf.local_variables_initializer())#使用tf.train.match_filenames_once(path)需要这句
    sess.run(tf.global_variables_initializer())
    xs, ys = sess.run([img_batch,label_batch])
    r1 = np.zeros((xs[1].shape[0], xs[1].shape[1]), dtype=xs[1].dtype)
    r2 = np.zeros((xs[1].shape[0], xs[1].shape[1]), dtype=xs[1].dtype)
    for i in range(len(ys)):
        print(xs[i].shape)
        #通道分离
        r1[:, :] = (xs[i])[:, :, 0]
        r2[:, :] = (xs[i])[:, :, 1]

        plt.subplot(2, 1, 1), plt.title('gray1')
        plt.imshow(r1, cmap='gray'), plt.axis('on')

        plt.subplot(2, 1, 2), plt.title('r1')
        plt.imshow(r2, cmap='gray'), plt.axis('on')

        plt.show()
# coord.request_stop()
# coord.join(threads)