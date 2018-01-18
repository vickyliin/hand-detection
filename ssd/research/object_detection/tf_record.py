
# coding: utf-8

# # Preparing inputs
# The inputs generation uses three step:
# 1. Pulling data
# 2. Formatting data to TFRecord files
# 3. Writting TFRecord files

# ## Imports

# In[1]:


import sys, os, io, hashlib, json
import scipy.io
import tensorflow as tf
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("research")
sys.path.append("research/object_detection")

from utils import dataset_util
from utils import label_map_util


# ## Variables
# 
# We know the only label will be "hand", so the label map file is ready-to-use in the repository.

# In[2]:


DATA_PATH = sys.argv[1]

# Test dataset
TEST_OUTPUT_FILENAME = 'research/object_detection/data/hands_test.record'

# Training dataset
TRAIN_OUTPUT_FILENAME = 'research/object_detection/data/hands_train.record'

# Validation dataset
VAL_OUTPUT_FILENAME = 'research/object_detection/data/hands_val.record'

# Real dataset
REAL_OUTPUT_FILENAME = 'research/object_detection/data/hands_real.record'

# The label map file with the "hand" label
LABEL_MAP_PATH = 'research/object_detection/data/hands_label_map.pbtxt'
label_map_dict = label_map_util.get_label_map_dict(LABEL_MAP_PATH)


# In[3]:


def create_tf_example(name, img_dir, ann_dir):

    IMG_FILENAME = '%s.png' % name
    ANN_FILENAME = 'label%s.json' % name[3:]
    IMG_FULL_PATH = os.path.join(img_dir, IMG_FILENAME)
    ANN_FULL_PATH = os.path.join(ann_dir, ANN_FILENAME)

    with tf.gfile.GFile(IMG_FULL_PATH, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'PNG':
        raise ValueError('Image format not PNG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    js = json.load(open(ANN_FULL_PATH, 'r'))
    for label in js['bbox'].keys():
        text_label = label + 'hand'
        x_0, y_0, x_1, y_1 = js['bbox'][label]
        x_0, x_1 = max(x_0, 0), max(x_1, 0)
        y_0, y_1 = max(y_0, 0), max(y_1, 0)
        x_0, x_1 = min(x_0, width), min(x_1, width)
        y_0, y_1 = min(y_0, height), min(y_1, height)
        
        if x_1 > x_0 and y_1 > y_0:
            # print(x_0, x_1, y_0, y_1, text_label)
            # print(x_0, x_1, y_0, y_1, width, height, IMG_FILENAME)
            xmin.append(float(x_0) / width)
            ymin.append(float(y_0) / height)
            xmax.append(float(x_1) / width)
            ymax.append(float(y_1) / height)
            classes_text.append(text_label.encode('utf-8'))
            classes.append(label_map_dict[text_label])
            truncated.append(0)
            poses.append('Frontal'.encode('utf-8'))
            difficult_obj.append(0)
            
        
    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
              IMG_FILENAME.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(
              IMG_FILENAME.encode('utf-8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('png'.encode('utf-8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
      }))


# ## Writting TFRecord files

# In[11]:


def create_tf_record(data_path, output_filename):
    writer = tf.python_io.TFRecordWriter(output_filename)
    print('Generating %s file...' % output_filename)
    if 'val' in output_filename:
        conditions = ['air', 'book']
    elif 'test' in output_filename:
        conditions = ['s009']                 
    elif 'real' in output_filename:
        conditions = ['air', 'book']
    else:
        conditions = ['s00{}'.format(i) for i in range(0, 9)]
    
    for cur_path in os.listdir(data_path):
        if 'DeepQ' in cur_path and not 'DeepQ-Mask-Arm' in cur_path:
            data_dir = os.path.join(data_path, cur_path, 'data')
            #print('\t{}'.format(data_dir))
            for sub_dir in os.listdir(data_dir):
                if any([True if cond in sub_dir else False for cond in conditions]):
                    print('\tCollecting images in : ', os.path.join(data_dir, sub_dir))
                    img_dir = os.path.join(data_dir, sub_dir, 'img')
                    ann_dir = os.path.join(data_dir, sub_dir, 'label')
                    for f in os.listdir(img_dir):
                        if '.png' in f:
                            if 'real' in output_filename:
                                if any([True if cond in f else False for cond in ['{0:05d}'.format(i) for i in range(1, 200, 40)]]):
                                    # print(f)
                                    img_name = f.split('.')[0]
                                    tf_example = create_tf_example(img_name, img_dir, ann_dir)
                                    writer.write(tf_example.SerializeToString())
                            if 'val' in output_filename:
                                if not any([True if cond in f else False for cond in ['{0:05d}'.format(i) for i in range(1, 200, 40)]]):
                                    # print(f)
                                    img_name = f.split('.')[0]
                                    tf_example = create_tf_example(img_name, img_dir, ann_dir)
                                    writer.write(tf_example.SerializeToString())
                            else:
                                img_name = f.split('.')[0]
                                tf_example = create_tf_example(img_name, img_dir, ann_dir)
                                writer.write(tf_example.SerializeToString())
    writer.close()
    print('%s written.' % output_filename)


# In[5]:


create_tf_record(DATA_PATH, REAL_OUTPUT_FILENAME)
create_tf_record(DATA_PATH, VAL_OUTPUT_FILENAME)
create_tf_record(DATA_PATH, TRAIN_OUTPUT_FILENAME)
create_tf_record(DATA_PATH, TEST_OUTPUT_FILENAME)

