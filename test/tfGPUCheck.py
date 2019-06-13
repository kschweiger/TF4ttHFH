import os
cudaDevices = os.environ['CUDA_VISIBLE_DEVICES']
print("Found CUDA_VISIBLE_DEVICES: %s"%cudaDevices)
if "," in cudaDevices:
    cudaDevices = cudaDevices.split(",")
else:
    cudaDevices = [cudaDevices]
    
thisDevice = '/gpu:'+str(cudaDevices[0])
print("Will use Device - %s"%thisDevice)

import tensorflow as tf

with tf.device(thisDevice):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
