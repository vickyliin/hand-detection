
# coding: utf-8

# In[ ]:


import os, sys
import numpy as np
import judger_hand
import tensorflow as tf
from PIL import Image


# In[ ]:


from object_detection.utils import label_map_util


# In[ ]:


# In[ ]:


# Model dir
PATH_TO_MODEL = 'htc_real'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join('research/object_detection', PATH_TO_MODEL, 'output/frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('research/object_detection', 'data', 'hands_label_map.pbtxt')

NUM_CLASSES = 2


# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map

# In[ ]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[ ]:


imgs = judger_hand.get_file_names()


# In[ ]:


IMAGE_SIZE = (12, 8)


# In[ ]:


def read_image_by_filename(img_path):
    img = Image.open(img_path)
    return np.array(img)


# In[ ]:


f = judger_hand.get_output_file_object()
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
         # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        for img in imgs:
            image = Image.open(img)
            image_np = np.array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            """
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()
            """

            for box, score, cls in zip(boxes[0][:], scores[0][:], classes[0][:]):
                x0 = int(box[1] * image.width)
                x1 = int(box[3] * image.width)
                y0 = int(box[0] * image.height)
                y1 = int(box[2] * image.height)
                result = '{} {} {} {} {} {} {}\n'.format(img, x0, y0, x1, y1, int(cls - 1), score)
                result = result.encode('utf-8')
                f.write(result)
        score, err = judger_hand.judge()
        if err is not None:  # in case we failed to judge your submission
            print(err)

