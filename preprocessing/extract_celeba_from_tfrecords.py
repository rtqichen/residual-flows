import numpy as np
import tensorflow as tf
import torch

sess = tf.InteractiveSession()

train_imgs = []

print('Reading from training set...', flush=True)
for i in range(120):
    tfr = 'data/celebahq/celeba-tfr/train/train-r08-s-{:04d}-of-0120.tfrecords'.format(i)
    print(tfr, flush=True)

    record_iterator = tf.python_io.tf_record_iterator(tfr)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        image_bytes = example.features.feature['data'].bytes_list.value[0]

        img = tf.decode_raw(image_bytes, tf.uint8)
        img = tf.reshape(img, [256, 256, 3])
        img = img.eval()

        train_imgs.append(img)

train_imgs = np.stack(train_imgs)
train_imgs = torch.tensor(train_imgs).permute(0, 3, 1, 2)
torch.save(train_imgs, 'data/celebahq/celeba256_train.pth')

validation_imgs = []
for i in range(40):
    tfr = 'data/celebahq/celeba-tfr/validation/validation-r08-s-{:04d}-of-0040.tfrecords'.format(i)
    print(tfr, flush=True)

    record_iterator = tf.python_io.tf_record_iterator(tfr)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        image_bytes = example.features.feature['data'].bytes_list.value[0]

        img = tf.decode_raw(image_bytes, tf.uint8)
        img = tf.reshape(img, [256, 256, 3])
        img = img.eval()

        validation_imgs.append(img)

validation_imgs = np.stack(validation_imgs)
validation_imgs = torch.tensor(validation_imgs).permute(0, 3, 1, 2)
torch.save(validation_imgs, 'data/celebahq/celeba256_validation.pth')
