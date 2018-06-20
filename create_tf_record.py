
import pandas as pd
import tensorflow as tf
import os
import io
import logging

from object_detection.utils import dataset_util


flags = tf.app.flags
# Arg 1 : Format ( arg, default, help msg)
flags.DEFINE_string('image_dir', '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/BirdDataset/CUB_200_2011/images/' ,
                                'Path to the image directory')
# Arg 2 : Format ( arg, default, help msg)
flags.DEFINE_string('output_dir', '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/birdDetection/detectionModel/data/' ,
                                'Path to output directory')
# Arg 3 : Format ( arg, default, help msg)
flags.DEFINE_string('csvData_dir', '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/birdDetection/random.csv' ,
                                'Path to data directory')
flags.DEFINE_string('status', 'val' ,
                                'Type of file, train or test?')
FLAGS = flags.FLAGS


SETS = ['train', 'val']

def create_tf_example(example, path):

    height = int( example["img_height"] ) # Image height
    width =  int ( example["img_width"] ) # Image width
    filename = example["img_name"] # Filename of the image. Empty if image is not from file
    img_path = os.path.join(path, filename)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    # Why we do this don't really know.
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    encoded_image_data =  encoded_jpg_io # Encoded image bytes

    image_format = 'jpeg'.encode('utf8') # or b'png'


    classes = []
    xmins = [example["x"] / width] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [(example["x"]+ example["width"])/width] # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [example["y"] / height] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [(example["y"]+ example["height"]) / height] # List of normalized bottom y coordinates in bounding box
    # (1 per box) :
    classes_text = ["bird".encode('utf8')] # List of string class name of bounding box (1 per box)
    classes = classes.append(1) # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
  logging.info('Reading from bird dataset')

  # If the argument does not really work
  if FLAGS.status not in SETS:
      raise ValueError('set must be in : {}'.format(SETS))

  output_path = FLAGS.output_dir
  output_file = output_path + str(FLAGS.status) + '.record'
  writer = tf.python_io.TFRecordWriter(output_file)
  # TODO(user): Write code to read in your dataset to examples variable
  # Path to read the images - Set as default argument on top
  # Defualt argument = '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/BirdDataset/CUB_200_2011/images/'
  path = FLAGS.image_dir
  # path to the file having all image information
  # Default Path '/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/birdDetection/random.csv'

  datasetFile = FLAGS.csvData_dir
  dataBase = pd.read_csv(datasetFile)

  if FLAGS.status is 'train':
      status_id = 1
      print("Training ")
  if FLAGS.status is 'val':
      status_id = 0
      print("Evaluation ")

  counter = 0
  for i in range(len(dataBase.index)):
      newFormat = dataBase.iloc[i,:]
      # for example in examples:
      if int(newFormat["train_status"]) == status_id :
          tf_example = create_tf_example(newFormat, path)
          counter = counter + 1
          writer.write(tf_example.SerializeToString())

  print("\n Total Val" , counter , ' Status : ', FLAGS.status)

  writer.close()
  print("\n Finishing Process!! : " + str(FLAGS.status))


# This allows the module to be executed on its own account, without being imported from another file.
if __name__ == '__main__':
    tf.app.run()
else:
    print('File called out by external sources')
