from tf_records import DataLoad
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.contrib.eager as tfe

LOGDIR = 'one'
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

dataset = DataLoad('./HAM10000/data.tfrecords', EPOCHS, BATCH_SIZE).return_dataset()
   
print(dataset)
# for i, batch in enumerate(dataset):
#     print(batch['image'], batch['image'].shape)
#     img = batch['image']
#     import numpy as np
#     print(img.shape)
#     img = np.reshape(img, (-1, 299, 299, 3))
#     import matplotlib.pyplot as plt
#     print('sss')
#     plt.imshow(img[0])
#     plt.show()
#     break