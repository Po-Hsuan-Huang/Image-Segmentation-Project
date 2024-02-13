#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:53:30 2019

@author: pohsuanh
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from glob import glob as glob
import pickle
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()


root_dir = '/home/pohsuanh/Documents/Lectures/CSCI699/hw2/'
    
os.chdir(root_dir)

if 'train_data' not in locals():

    img_path = os.path.join(root_dir, 'hw2_data','images')
    
    label_path = os.path.join(root_dir, 'hw2_data', 'segmentation')
    
def plot_images(dataset, n_images, samples_per_image):


    output = np.zeros((192 * n_images, 192 * samples_per_image, 3))

    row = 0
    for images in tfe.Iterator(dataset.repeat(samples_per_image).batch(n_images)):
        try :
            output[:, row*192:(row+1)*192,:] = np.vstack(images.numpy())
        except ValueError : 
            break

        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()
    
    
def load_and_preprocess(paths):

    frames = []
    
    for path in paths :
        v = tf.io.read_file(path)
        img_tensor = tf.image.decode_jpeg(v, channels = 3)
        img_final = tf.image.resize_images(img_tensor, [192, 192])
        img_final = img_final/255.0
        frames.append(img_final)            
        
    tf_data = tf.convert_to_tensor(frames)
    
    return tf_data


def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def augment(Dataset, augmentations):
    """ Apply augmentations
        
    Args:
        
        Dataset: Tensorflow Dataset
        augmentations: list of flags
        
    Returns :
        
        Tensorflow Dataset
    
    """
    # Apply an augmentation only in 25% of the cases.
    for f in augmentations:
        Dataset.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x), num_parallel_calls=4)

    return Dataset

### Laad data from .png files
    
tensor_imgs = load_and_preprocess(sorted(glob(os.path.join(img_path,'*.jpg'))))
tensor_targets = load_and_preprocess(sorted(glob(os.path.join(label_path,'*.png'))))   

data_imgs = tf.data.Dataset.from_tensor_slices(tensor_imgs)
data_targets = tf.data.Dataset.from_tensor_slices(tensor_targets)

# Data augmentation

augmentations = [flip, color]

data_imgs  = augment(data_imgs, augmentations)


plot_images(data_imgs, 6, 6)

# Split training set and test set

train_fraction = 0.8
validation_fraction = 0.1
test_fraction = 0.1

train_size = int(tensor_imgs.shape[3] * train_fraction)

val_size = int(tensor_imgs.shape[3] * validation_fraction)

test_size = tensor_imgs.shape[3] - train_size - val_size

train_imgs, val_imgs, test_imgs = data_imgs.take(train_size), data_imgs.skip(train_size).take(val_size), data_imgs.skip(train_size + val_size)
train_targets, val_targets, test_targets = data_targets.take(train_size), data_targets.skip(train_size).take(val_size), data_imgs.skip(train_size + val_size)

tf.compat.v1.disable_eager_execution()

## Build Graph :


## Execute Graph :

with tf.Session() as sess :
    
    train_iterator = train_imgs.make_one_shot_iterator().repeat().shuffle(100).batches(20)
    train_iterator_handle = sess.run(train_iterator.string_handle())
    
    val_iterator = val_imgs.make_one_shot_iterator().repeat().shuffle(100).batches(20)
    train_iterator_handle = sess.run(train_iterator.string_handle())
    
    test_iterator = test_imgs.make_one_shot_iterator().repeat().shuffle(100).batches(20)
    test_iterator_handle = sess.run(test_iterator.string_handle())




train_loss = sess.run(loss, feed_dict={handle: train_iterator_handle})
test_loss = sess.run(loss, feed_dict={handle: test_iterator_handle})
features_train, labels_train, features_test, labels_test = tf.train_split(features,
                                                                            labels,
                                                                            frac=.1)
