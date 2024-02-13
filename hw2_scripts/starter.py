import os
import sys
from model import Deeplabv3 
from absl import flags, app
import numpy as np
import tensorflow as tf
from model import relu6

#flags.DEFINE_string('data_dir','hw2_data','Directiory containing sub-directories tf-segmentation and images.')
#flags.DEFINE_string('model_dir','hw2_models','Directiory containing saved model weights and configs files')

#FLAGS = flags.FLAGS



def get_filename_data_readers(image_ids_file, get_labels=False,
                              x_jpg_dir=None, y_png_dir=None):
  """Given image IDs file, returns Datasets, which generate image paths.

  The goal of this function is to convert from image IDs to image paths.
  Specifically, the return type should be:
    if get_labels == False, return type should be tf.data.Dataset.
    Otherwise, return type should be pair (tf.data.Dataset, tf.data.Dataset).
  In both cases, the Dataset objects should be "not batched".

  For example, if the file contains 2 lines: "0000\n0001", then the returned
  dataset should give an iterator that when its tensor is ran, gives "0000" the
  first time and gives "0001" the second time.

  Args:
    image_ids_file: text with one image ID per line.
    get_labels: If set, returns 2 Datasets: the containing the image files (x)
      and the second containing the segmentation labels (y). If not, returns
      only the first argument.
    x_jpg_dir: Directory where each image lives. Specifically, image with
      ID "image1" will live on "x_jpg_dir/image1.jpg".
    y_png_dir: Directory where each segmentation mask lives. Specifically,
      image with ID "image1" will live on "x_png_dir/image1.png".
  
  Returns:
    instance of tf.data.Dataset, or pair of instances (if get_labels == True).
  """
  x_jpg_dir = x_jpg_dir or os.path.join(FLAGS.data_dir, 'images')
  y_png_dir = y_png_dir or os.path.join(FLAGS.data_dir, 'tf_segmentation')
  # TODO(student): Write code.
  
  with open(image_ids_file,'r') as f:
      img_ids = f.read().splitlines() 
      
  img_paths =[]  
  
  for f in img_ids : 
      
      img_paths.append( tf.cast(os.path.join( x_jpg_dir, f + '.jpg'), dtype = tf.string))
  
  if get_labels == False :
      
      data = tf.data.Dataset.from_tensor_slices(img_paths)
      
      return data
  
  elif get_labels == True :
      
      with open(image_ids_file,'r') as f:
          img_ids = f.read().splitlines() 
      
      label_paths = []
      
      for f in img_ids : 
          label_paths.append( tf.cast( os.path.join( y_png_dir, f + '.png'), dtype = tf.string))
          
      return tf.data.Dataset.from_tensor_slices(img_paths),tf.data.Dataset.from_tensor_slices(label_paths)

def decode_image_with_padding(im_file, decode_fn=tf.image.decode_jpeg,
                              channels=3, pad_upto=500):
  """Reads an image, decodes, and pads its spatial dimensions, all in TensorFlow

  Args:
    im_file: tf.string tensor, containing path to image file.
    decode_fn: Tensorflow function for converting
    channels: Image channels to decode. For data (x), set to 3 channels (i.e. RGB).
      For labels (segmentation masks), set to 1, because other 2 channels contain
      identical information.
    pad_upto: Number of pixels to pad to.

  Returns:
    Pair of Tensors:
      The first must be tf.int vector with 2 entries: containing the original height
        and width of the image.
      The second must be a tf.int matrix with size (pad_upto, pad_upto, 3)
        i.e. the contents of the image, with zero-padding.
  """
  # TODO(student): Write code.
  f = tf.io.read_file(im_file)
  img_tensor = decode_fn(f, channels = channels)
  shape = tf.shape(img_tensor)[:2]
  img_final = tf.image.pad_to_bounding_box(img_tensor, 0, 0, pad_upto, pad_upto)
  img_final = tf.cast(img_final, tf.int32)
  
  return (shape, img_final)
  


def make_loss_mask(shapes):
  """Given tf.int Tensor matrix with shape [N, 2], make N 2D binary masks.
  
  These binary masks will be used "to mask the loss". Specifically, if the
  image is shaped as (300 x 400) and therefore so its labels, we only want
  to penalize the model for misclassifying within the image boundary (300 x 400)
  and ignore values outside (e.g. at pixel [350, 380]).

  Args:
    shapes: tf.int Tensor with shape [N, 2]. Entry shapes[i] will be a vector:
      [image height, image width].

  Returns:
    tf.float32 mask of shape [N, 500, 500], with mask[i, h, w] set to 1.0
    iff shapes[i, 0] < h and shapes[i, 1] < w.
  """
  # TODO(student): Write code.
  batch_size = np.shape(shapes)[0]
  masks = np.zeros([batch_size, 500, 500])
  
  for i, shape in enumerate(shapes):
      h, w = shape[0], shape[1]
      masks[ i,: h,: w] = np.ones([ h, w])
      
  masks =tf.convert_to_tensor(masks, dtype=tf.float32)    
      
  return masks



def read_image_pair_with_padding(x_im_file, y_im_file, pad_upto=500):
  """Reads image pair (image & segmentation). You might find it useful.

  It only works properly, if you implemented `decode_image_with_padding`. If you
  do not find this function useful, ignore it.
  not have to use this function, if you do not find it useful.
  
  Args:
    x_im_file: Full path to jpg image to be segmented.
    y_im_file: Full path to png image, containing ground-truth segmentation.
    pad_upto: The padding of the images.

  Returns:
    tuple of tensors with 3 entries:
      int tensor of 2-dimensions.
  """
  shape, im_x = decode_image_with_padding(x_im_file)
  _    , im_y = decode_image_with_padding(y_im_file, tf.image.decode_png, channels=1)
  return shape, im_x, im_y



class SegmentationModel:
  """Class that can segment images into 21 classes (class 0 being background).

  You must implement the following in this class:
  + load()
  + predict()
  which will be called by the auto-grader

  + train()
  + save()
  which will NOT be called by the auto-grader, but you must implement them for
  repeatability. After the grading period, we will re-train all models to make
  sure their present training will get their results.
  """

  def __init__(self):
    # 0: background class, {1, .., 20} are for class labels.
    self.num_classes = 21
    self.batch_size = 100  # You can change or remove
    self.num_iter = 2000
    self.model_dir = '/Users/pohsuanhuang/Documents/Lectures/CSCI699/hw2/hw2_data/'
    
  def save(self, model_dir):
    """Saves model parameters to disk."""
    # Save JSON config to disk
    json_config = self.model.to_json()
    with open('model_config.json', 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk
    self.model.save_weights('path_to_my_weights.h5')

    print("Model saved in path: %s" % model_dir)



  def load(self, model_dir):
    """Restores parameters of model from `model_in_file`."""
    # Reload the model from the 2 files we saved
    with open(os.path.join(model_dir,'model_config.json')) as json_file:
        json_config = json_file.read()
    new_model = tf.keras.models.model_from_json(json_config)
    new_model.load_weights(os.path.join(model_dir,'path_to_my_weights.h5'))

    print("Model laoded from path: %s" % model_dir)


  def predict(self, images):
    """Predicts segments of some images.

    This method WILL BE CALLED by auto-grader. Please do not change signature of
    function [though adding optional arguments is fine].

    Args:
      images: List of images. All will be padded to 500x500 by autograder. The
        list can be a primitive list or a numpy array. To go from the former to
        the latter, just wrap as np.array(images).

    Returns:
      List of predictions (or numpy array). If np array, must be of shape:
      N x 500 x 500 x 21, where N == len(images). Our evaluation script will
      take the argmax on the last column for accuracy (and might calculate AUC).
    """
    
    softmax = self.model(images)
    predictions = tf.multiply( softmax)

    return np.asarray(predictions)

  def train(self, train_ids_file):
    """Trains the model.
    
    This method WILL BE CALLED by our scripts, after submission period. Please
    do not add required arguments. Feel free to completely erase its body.

    Args:
      train_ids_file: file containing image IDs.
    """
    # TODO(student): Feel free to remove if you do not use. 
    " Load data "
#    data_dir = '/home/pohsuanh/Documents/Lectures/CSCI699/hw2/hw2_data'
    data_dir = '/Users/pohsuanhuang/Documents/Lectures/CSCI699/hw2/hw2_data/'
    img_path,label_path = get_filename_data_readers(train_ids_file, x_jpg_dir = os.path.join(data_dir,'images'),
                                                       y_png_dir = os.path.join(data_dir, 'tf_segmentation'),
                                                       get_labels= True)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    Data = tf.data.Dataset.zip((img_path,label_path)).map(read_image_pair_with_padding,num_parallel_calls=AUTOTUNE)
    DATASET_SIZE = 10
    train_size = int(0.8 * DATASET_SIZE)
    test_size = int(0.2 * DATASET_SIZE)

    train_ds = Data.take(train_size)
    test_ds = Data.skip(train_size)
    test_ds = Data.take(test_size)
    "shuffle repeat batch"
    train_ds = train_ds.shuffle(1000).repeat().batch(2).prefetch(buffer_size=AUTOTUNE)
    
    "build model"
    self.model = self.build_train_model()
#    self.model.summary()
    L2_rate = tf.constant(0.01, tf.float32)
        
    l2_loss = tf.constant(0.0, tf.float32) #tf.add_n(self.model.losses) 
    loss_object =tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    # Configure the trainable Op (for TRAIN mode)
                
    optimizer = tf.keras.optimizers.Adam()
    
    @tf.function
    def train_step(images, labels, loss_masks):
      with tf.GradientTape() as tape:
        images = tf.cast(images, tf.float32)
#        softmax = self.model(images)
#        predictions = tf.multiply( softmax, loss_masks)
        predictions = self.model(images)
        print("prediction:",predictions)

        loss = loss_object(labels , predictions) #+ L2_rate * tf.reduce_mean(l2_loss)
    
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
        train_loss(loss)
        train_accuracy(labels, predictions)
        
    @tf.function
    def test_step(images, labels):
#        softmax = self.model(images)
#        predictions = tf.multiply( softmax, loss_masks)
        images = tf.cast(images, tf.float32)
        predictions = self.model(images)
        loss = loss_object(labels , predictions) #+ L2_rate * tf.reduce_mean(l2_loss)
    
        test_loss(loss)
        test_accuracy(labels, predictions)
        
    EPOCHS = 5
    
    self.model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy',
                       metrics =['accuracy'])
    
    self.model.summary()
    
    for epoch in range(EPOCHS):
      for it, (shapes, images, segs) in enumerate(train_ds): 
        loss_masks = make_loss_mask(shapes)
        loss_masks = tf.expand_dims(loss_masks, 3)
        loss_masks_all_classes = tf.keras.backend.repeat_elements(loss_masks,rep = 21, axis = 3)
        labels = tf.multiply(loss_masks_all_classes, loss_masks_all_classes)
        self.model.fit(images,labels)

    for epoch in range(EPOCHS):
      for it, (shapes, images, labels) in enumerate(train_ds):
        loss_masks = make_loss_mask(shapes)
        loss_masks = tf.expand_dims(loss_masks, 3)
        loss_masks_all_classes = tf.keras.backend.repeat_elements(loss_masks,rep = 21, axis = 3)
        train_step(images, labels, loss_masks_all_classes)
        template = 'Iteration {}, Loss: {}, Accuracy: {}'
        print(template.format(it+1,
                            train_loss.result(),
                            train_accuracy.result()*100))
    
      for shape, test_images, test_labels in test_ds:
        loss_masks = make_loss_mask(shapes)
        loss_masks = tf.expand_dims(loss_masks, 3)
        loss_masks_all_classes = tf.keras.backend.repeat_elements(loss_masks,rep = 21, axis = 3)
        test_step(test_images, test_labels, loss_masks_all_classes)
    
      template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
      print(template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            test_loss.result(),
                            test_accuracy.result()*100))
    
      # Reset the metrics for the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()


  # TODO(student): Feel free to remove if you do not use.
  def build_train_model(self):
      
     return Deeplabv3(input_shape=(500, 500, 3), classes=21, weights= 'pascal_voc',activation='softmax')  
         

def main(argv):
  # TODO(student): Feel free to write whatever here. Our auto-grader will *not*
  # invoke this at all.
  
  model = SegmentationModel()
#  model.train((os.path.join('/home/pohsuanh/Documents/Lectures/CSCI699/hw2/hw2_data/','test.txt')))
  model.train((os.path.join('/Users/pohsuanhuang/Documents/Lectures/CSCI699/hw2/hw2_data/','test.txt')))


if __name__ == '__main__':
  print('run')
  app.run(main)

#  flags.DEFINE_string('data_dir', '/home/pohsuanh/Documents/Lectures/CSCI699/hw2/hw2_data','Directiory containing sub-directories tf-segmentation and images.')  

#  FLAGS = flags.FLAGS
#  data_dir = '/home/pohsuanh/Documents/Lectures/CSCI699/hw2/hw2_data'
#  if not os.path.exists(FLAGS.data_dir):
#    sys.stderr.write('Error: Data directory not found: %s\n' % FLAGS.data_dir)
    
#  model = SegmentationModel()
  
#  model.model_dir = '/home/pohsuanh/Documents/Lectures/CSCI699/hw2/hw2_data/ckpts/'
  
#  model.train(os.path.join('/home/pohsuanh/Documents/Lectures/CSCI699/hw2/hw2_data/','img_id.txt'))
#  if 'pair' not in locals():
#      print('pair')
#      im_path, seg_path = get_filename_data_readers(os.path.join(data_dir,'img_id.txt'),
#                              x_jpg_dir = os.path.join(data_dir,'images'),
#                              y_png_dir = os.path.join(data_dir, 'tf_segmentation'),
#                              get_labels=True)
#      
#      
#      
##      list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#
#
#  shapes = []
#  images = []
#  segments = []
#  (a,b) = pair
#  for x,y in zip(a.take(5),b.take(5)):
#      
#      shape, imx, imy = read_image_pair_with_padding(x, y, pad_upto=500)
#
#      shapes.append(shape)
#      images.append(imx)
#      segments.append(imy)
#  masks = make_loss_mask(shapes)
#  
#  
#  
  
  

#  (img_path,label_path) = get_filename_data_readers(os.path.join(data_dir,'img_id.txt'),
#                                                       x_jpg_dir = os.path.join(data_dir,'images'),
#                                                       y_png_dir = os.path.join(data_dir, 'tf_segmentation'),
#                                                       get_labels= True)
#  data = tf.data.Dataset.from_tensors((img_path, label_path))
#    
#  data.map(read_image_pair_with_padding, num_parallel_calls=AUTOTUNE) # conta
#
#      
#  
#  
  
