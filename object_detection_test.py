
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.
# 
# 
# # Testing Script
# The idea is to create a script which takes a bunch of images from the given folder and then tests them through a network to see the result. 

# # Imports

# In[26]:


import numpy as np
import os
import sys
import tensorflow as tf
import glob
import cv2


from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# ## Env setup
# ## Object detection imports
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# Flags for calling the independent module
flags = tf.app.flags
flags.DEFINE_string('inputType', 'img' , 'type of data : img = image or vid = video')
FLAGS = flags.FLAGS



# ## Helper code
# The code is transforming the image to numpy array, easier to pass through the network
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  print(image.size)
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# This function is running each image through the model and get the detection output
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension.
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            sess.close()
    return output_dict


def main(_):

    # The model path and the associate files will remain same
    MODEL_PATH = '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/birdDetection/detectionModel/'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_PATH + 'outputModel/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = MODEL_PATH + os.path.join('data', 'label_map.pbtxt')
    NUM_CLASSES = 1


    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`,
    # we know that this corresponds to `airplane`.  Here we use internal utility functions,
    # but anything that returns a dictionary mapping integers to appropriate string labels would
    # be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # # Detection
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    #PATH_TO_TEST_IMAGES_DIR = MODEL_PATH + 'gazelle_images'


    # If a bunch of images are given to the algorithm
    if FLAGS.inputType is 'img':

        PATH_TO_TEST_IMAGES_DIR = '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/BirdDataset/CUB_200_2011/images/*/'
        PATH_TO_SAVE_TEST_IMAGES_DIR = '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/BirdDataset/testResult/'

        TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR,'*.jpg'), recursive = True)

        # for image_path in TEST_IMAGE_PATHS:
        #     image = image_path
        #     name  = os.path.basename(image_path)
        #     namewoext = os.path.splitext(name)[0]

        # Go through each image and send it through the network
        for image_path in TEST_IMAGE_PATHS:
          image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = load_image_into_numpy_array(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3] ---> Done in the function for inference
          # image_np_expanded = np.expand_dims(image_np, axis=0)

          # Actual detection.
          output_dict = run_inference_for_single_image(image_np, detection_graph)
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=8)

          name  = os.path.basename(image_path)

          save_image_as = os.path.join(PATH_TO_SAVE_TEST_IMAGES_DIR, name)
          #plt.savefig(save_image_as)
          testImage = Image.fromarray(image_np)
          testImage.save(save_image_as)


    elif FLAGS.inputType is 'vid':
        imagePath = '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/ObjDet_TestingScript/MateFeeding.avi'
        save_path = '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/ObjDet_TestingScript/testResults/birds'

        cap = cv2.VideoCapture(imagePath)
        # Ideally to be stored as video output
        #outputVideo = cv2.VideoWriter('testBirds.mp4',0x00000020, 30.0, (1280,720), True)

        counter = 0
        while (cap.isOpened()):

            ret, frame = cap.read()

            if frame is None:
                print('There was no frame to read')
                break

            #----Skip----- this step is the image is loaded with opecv, since it is numpy compatible
            # image_np = load_image_into_numpy_array(frame)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            #image_np_expanded = np.expand_dims(frame, axis=0)


            output_dict = run_inference_for_single_image(frame, detection_graph) #frame
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)


            name = save_path + str(counter) + '.jpg'
            counter = counter + 100

            cap.set(cv2.CAP_PROP_POS_FRAMES, counter)

            # Display and write images at default location
            cv2.imshow("testWindow", frame)
            cv2.imwrite(name, frame)

            key = cv2.waitKey(10)
            if (key == ord('q')):
                # cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
        #outputVideo.release() # Release the video writer

    else:
        print("Wrong input")
        raise ValueError('set must be in : vid or img')


# The function is called only if the file is called on its own, this enables using files independently
if __name__ == '__main__':
    print('Invoked from console!!')
    tf.app.run()
else:
    print('File called out by external sources/program.')



