#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:42:11 2018

@author: pohsuanh


Fully Covolutional Network FCN-32s. 

FCN-32s network is based on VGG-16

"""
import tensorflow.compat.v2.keras as keras
import tensorflow.compat.v1 as tf
import numpy as np
    
def conv2d(inputs, filters, kernel_size, name, kernel_regularizer, trainable,  activation = 'relu'):
    
    x = tf.layers.conv2d(inputs, filters, kernel_size,
                          activation= 'linear',
                          padding='valid',
                          name= name,
                          kernel_regularizer= kernel_regularizer,
                          trainable = trainable)
    
    x2 = tf.layers.batch_normalization(x)
    
    if activation == 'relu':
    
        x3 = tf.nn.relu(x2)
    
    if activation == 'sigmoid':
        
        x3 = tf.nn.sigmoid(x2)
    
    return x3


def dropout( x, rate ,  name, training = True ):
    
    if False :
    
        y = tf.compat.v2.keras.layers.SpatialDropout2D(rate = 0.4)
        
    else :
    
        y =tf.layers.dropout(x, rate = 0.4, training = training , name =name)
    
    return y
    
    

def adapt_network_for_any_size_input():
    
    pass

class fcn_model(object):
    
    def __init__(self):
        
        tf.reset_default_graph()
        
        with tf.variable_scope("down_sampling", reuse = tf.AUTO_REUSE):
    
            L2 = keras.regularizers.l2(l=0.1)
                                
            self.trainable = True
            
            seed = 2019
            
            self.features = tf.placeholder(tf.float32,[None,500,500,3])
            
            self.labels = tf.placeholder(tf.float32,[None,500,500,21])
            
            self.loss_masks = tf.placeholder(tf.float32, self.labels.shape)
            
            self.c1 = conv2d(self.features, 64, (7, 7),
                              activation='relu',
                              name='block1_conv1',
                              kernel_regularizer= L2,
                              trainable = self.trainable)
            
            self.d1 = dropout(self.c1, rate = 0.4, training = self.trainable , name ='block1_dp1')

            self.c2 =  conv2d(self.d1, 64, (7, 7),
                              activation='relu',
                              name='block1_conv2',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
         
            self.d2 = dropout(self.c2, rate = 0.4,  training = self.trainable , name ='block2_dp2')
        
            self.p1 =  tf.layers.max_pooling2d(self.d2, (2, 2), strides=(2, 2), name='block1_pool')
            
            # Block 2
            self.c3 = conv2d(self.p1, 128, (5, 5),
                              activation='relu',
                              name='block2_conv1',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
            
            self.d3 =dropout(self.c3, rate = 0.4,  training = self.trainable , name ='block2_dp1')
        
            
            self.c4 = conv2d(self.d3, 128, (5, 5),
                              activation='relu',
                              name='block2_conv2',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
            
            self.d4 =dropout(self.c4, rate = 0.4,  training = self.trainable , name ='block2_dp2')
        
        
            self.p2 = tf.layers.max_pooling2d(self.d4,(2, 2), strides=(2,2), name='block2_pool')
            
            # Block 3
            self.c5 = conv2d (self.p2, 256, (3, 3),
                              activation='relu',
                              name='block3_conv1',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
            
            self.d5 =dropout(self.c5, rate = 0.4,  training = self.trainable , name ='block3_dp1')
        
            self.c6 = conv2d (self.d5, 256, (3, 3),
                              activation='relu',
                              name='block3_conv2',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
            
            self.d6 =dropout(self.c6, rate = 0.4,  training = self.trainable , name ='block3_dp2')
        
            
            self.c7 = conv2d (self.d6, 256, (3, 3),
                              activation='relu',
                              name='block3_conv3',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
            
            self.d7 =dropout(self.c7, rate = 0.4,  training = self.trainable , name ='block3_dp3')
        
            
            self.p3 = tf.layers.max_pooling2d(self.d7, (2, 2), strides=(2, 2), name='block3_pool')
            
            # Block 4
            self.c8 = conv2d (self.p3, 512, (3, 3),
                              activation='relu',
                              name='block4_conv1',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
            
            self.d8 =dropout(self.c8, rate = 0.4,  training = self.trainable , name ='block4_dp1')
        
            self.c9 = conv2d (self.d8, 512, (3, 3),
                              activation='relu',
                              name='block4_conv2',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
            
            self.d9 =dropout(self.c9, rate = 0.4,  training = self.trainable , name ='block4_dp2')
        
            self.c10 = conv2d (self.d9, 512, (3, 3),
                              activation='relu',
                              name='block4_conv3',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
            
            self.d10 =dropout(self.c10, rate = 0.4,  training = self.trainable , name ='block4_dp3')
        
            self.p4 = tf.layers.max_pooling2d(self.d10, (2, 2), strides=(2, 2), name='block4_pool')
            
            # Block 5
            self.c11 = conv2d (self.p4, 512, (2, 2),
                              activation='relu',
                              name='block5_conv1',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
            
            self.d11 =dropout(self.c11, rate = 0.4,  training = self.trainable , name ='block5_dp1')
            
            self.c12 = conv2d (self.d11, 512, (2, 2),
                              activation='relu',
                              name='block5_conv2',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
        
            self.d12 =dropout(self.c12, rate = 0.4,  training = self.trainable , name ='block5_dp2')
            
            self.c13 = conv2d (self.d12, 512, (2, 2),
                              activation='relu',
                              name='block5_conv3',
                              kernel_regularizer= L2,
                              trainable  = self.trainable)
        
            self.d13 =dropout(self.c13, rate = 0.4,  training = self.trainable , name ='block5_dp3')
        
            self.p5 = tf.layers.max_pooling2d(self.d13, (2, 2), strides=(2, 2), name='block5_pool')
            
            # Block 6
            
            self.c14 = conv2d(self.p5, 4096, (7,7), 
                                 activation='relu',
                                 name='block6_conv1',
                                 kernel_regularizer= L2,
                                 trainable  = self.trainable)
            
            self.d14 =dropout(self.c14, rate = 0.4,  training = self.trainable , name ='block6_dp1')
        
            self.c15 = conv2d(self.d14, 4096, (1,1),
                                 activation='relu',
                                 name='block6_conv2',
                                 kernel_regularizer= L2,
                                 trainable  = self.trainable)
        
            self.d15=dropout(self.c15, rate = 0.4,  training = self.trainable , name ='block6_dp2')
            
            self.c16 = conv2d(self.d15, 21, (1,1),
                                 activation='relu',
                                 name='block6_conv3',
                                 kernel_regularizer= L2,
                                 trainable  = self.trainable) 
        
            self.d16 =dropout(self.c16, rate = 0.4,  training = self.trainable , name ='block6_dp3')

        
        with tf.variable_scope("FC32", reuse = tf.AUTO_REUSE):
            
            self.deconv1 = tf.layers.conv2d_transpose(self.d16, 21,(16,16), strides=(4,4),
                                           activation='sigmoid',
                                           padding = 'same',
                                           name='block7_deconv1',
                                           kernel_regularizer= L2,
                                           trainable  = self.trainable)
            
            self.deconv2 = tf.layers.conv2d_transpose(self.deconv1, 21, (100,100), strides=(25,25),
                                           activation='sigmoid',
                                           padding='same',
                                           name='block7_deconv2',
                                           kernel_regularizer= L2,
                                           trainable  = self.trainable) 
        
            self.logit1 = tf.multiply( self.deconv2, self.loss_masks, name = 'block7_mul')
        
        with tf.variable_scope("FC16", reuse = tf.AUTO_REUSE):
            
            
            self.deconv3 = tf.layers.conv2d_transpose(self.d16, 21,(16,16), strides=(5,5),
                                           activation='sigmoid',
                                           padding = 'same',
                                           name='block8_deconv1',
                                           kernel_regularizer= L2,
                                           trainable  = self.trainable)
            
            self.cat1 = keras.layers.concatenate([self.deconv3, self.p4], axis = -1, name = 'block8_cat1')
            
            self.deconv4 = tf.layers.conv2d_transpose(self.cat1, 21, (64,64), strides=(20,20),
                                           activation='sigmoid',
                                           padding='same',
                                           name='block8_deconv2',
                                           kernel_regularizer= L2,
                                           trainable  = self.trainable) 
        
            self.logit2 = tf.multiply( self.deconv4, self.loss_masks, name = 'block8_mul')
            
        with tf.variable_scope("FC8", reuse = tf.AUTO_REUSE):
            
            self.c17 = conv2d(self.d7, 128, (7,7),
                                 activation='relu',
                                 kernel_regularizer= L2,
                                 trainable  = self.trainable, name = 'block9_conv1') 
            
            self.c18 = conv2d(self.c17, 21, (7,7),
                                 activation='relu',
                                 kernel_regularizer= L2,
                                 trainable  = self.trainable, name = 'block9_conv2') 
            
            
            self.deconv5 = tf.layers.conv2d_transpose(self.deconv3, 21,(16,16), strides=(4,4),
                                           activation='sigmoid',
                                           padding = 'same',
                                           name='block8_deconv1',
                                           kernel_regularizer= L2,
                                           trainable  = self.trainable)
            
            self.cat2 = keras.layers.concatenate([self.deconv5, self.c18], axis = -1, name = 'block9_cat1')
            
            self.deconv6 = tf.layers.conv2d_transpose(self.cat2, 21, (20,20), strides=(5,5),
                               activation='sigmoid',
                               padding='same',
                               name='block9_deconv1',
                               kernel_regularizer= L2,
                               trainable  = self.trainable) 
            
        
            self.logit3 = tf.multiply( self.deconv6, self.loss_masks, name= 'block9_mul')
            
    
        with tf.variable_scope("logits", reuse = tf.AUTO_REUSE):
        
            self.cat3 = keras.layers.concatenate([self.deconv2, self.deconv4, self.deconv2], axis = -1, name = 'block10_cat1')
            
            self.conv = conv2d(self.cat3, 21, (1,1),
                                 activation='relu',
                                 name='block10_conv1',
                                 kernel_regularizer= L2,
                                 trainable  = self.trainable)
            
            self.logit = tf.multiply( self.conv, self.loss_masks, name = 'block10_mul')
            
            
        
    def predict(self):

        
            # Do piself.xel-wise predictions :
            
            self.predictions = {
                    
              # Generate predictions (for PREDICT and EVAL mode)
              
              "classes": tf.argmax(input=tf.reshape(self.logit,(1,None)), axis=1).reshape(self.logit.shape),
              
              # Add `softmax_tensor` to the graph. It is used for PREDICT and by the logging_hook`.
              
              "probabilities": tf.nn.softmax(self.logit, name="softmax_tensor")
        
              }
            
        
#            if mode == tf.estimator.ModeKeys.PREDICT:
              
            return self.predictions
        
    def train(self):
        
        # Calculate Loss (for both TRAIN and EVAL modes)
        # Homework requires tf.nn.sigmoid_cross_entropy_with_logits()
        
        self.loss_reg_L2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        
        self.L2_rate = 0.01
        
        # reduce dim from one-hot encoding to scalar encodin
        
    
        self.loss_object =tf.keras.losses.SparseCategoricalCrossentropy( from_logits = True)
                
        self.loss = tf.reduce_mean(self.loss_object(y_true=self.labels, y_pred= self.logit)) + self.L2_rate * tf.reduce_mean(self.loss_reg_L2)
        
        # Configure the trainable Op (for TRAIN mode)
        
#        if mode == tf.estimator.ModeKeys.TRAIN:
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        
        self.train_op = self.optimizer.minimize(loss=self.loss, global_step = tf.train.get_global_step())
        
        return  self.train_op
        
    def evaluate(self):
        
        # Add evaluation metrics (for EVAL mode)
#        if mode == tf.estimator.ModeKeys.TRAIN:
        
        self.tp = tf.metrics.true_positives(self.labels, self.predictions['classes'])
        
        self.fp = tf.metrics.false_positives(self.labels, self.predictions['classes'])
        
        self.fn = tf.metrics.false_negatives(self.labels, self.predictions['classes'])
        
        self.eval_metric_ops = {"IoU": self.tp/(self.tp + self.fp + self.fn)}
        
        return self.mode, self.loss, self.eval_metric_ops
    
    
'''
if __name__ == "__main__":
    
    root_dir = '/home/pohsuanh/Documents/Computer_Vision/HW6'

    # Load training and eval data
  
    train_data, eval_data, test_data = data_load.load()
    
    # Construct model
    pic = np.random.randint((test_data['x']).shape[0])
    
    image_sample = test_data['x'][pic]
    
    label_sample = test_data['y'][pic]
    
    image_sample = tf.Session().run(image_sample)
    
    label_sample = tf.Session().run(label_sample)
    
    plt.figure(figsize=(20,40))
    plt.title('data')
    plt.imshow(image_sample)
    
    plt.figure(figsize =(20,40))
    plt.title('gt')
    plt.imshow(label_sample)
        
    # Create the Estimator
    
    fcn_segmentor = tf.estimator.Estimator(
    
    model_fn=fcn_model_fn, model_dir=root_dir)
   
    # Set up logging for predictions

    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
                                   tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data['x'],
        y=train_data['y'],
        batch_size=1,
        num_epochs=None, # number of epochs to iterate over data. If None will run forever.
        shuffle=True)
   
    fcn_segmentor.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
   
    # Evaluate the model and print results
   
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data['x'],
        y=eval_data['y'],
        num_epochs=1,
        shuffle=False)
   
    eval_results = fcn_segmentor.evaluate(input_fn=eval_input_fn)
   
    print(eval_results)

'''