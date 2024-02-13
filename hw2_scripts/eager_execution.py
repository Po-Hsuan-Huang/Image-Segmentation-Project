#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:13:22 2019

@author: pohsuanh
"""

import tensorflow.contrib.eager as tfe

dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([50, 10,10]))
dataset = dataset.repeat(6).batch(6)
for batch in tfe.Iterator(dataset):
     print(batch.shape)