import pdb
import os
from glob import glob

class non_max_suppression:
    def __init__(self, defaults={}):
        self.__name__ = type(self).__name__
        self.defaults = {k: v for k, v in defaults.items() if v is not None}
    def __call__(self, detections, **kwargs):
        import tensorflow as tf
        from object_detection.core.post_processing import batch_multiclass_non_max_suppression as nms
        kwargs.update(self.defaults)
        boxes, scores = detections
        boxes = tf.expand_dims(boxes, 2)
        boxes, scores, classes = nms(boxes, scores, **kwargs)[:3]
        scores_classes = tf.stack([scores, classes], axis=-1)
        return tf.concat([boxes, scores_classes], axis=-1)

def build_model(backbone, nms_args, weights=None):
    from keras.layers import Input, Lambda
    from keras.optimizers import Adam
    from keras.models import Model
    from keras_retinanet.models.resnet import resnet_retinanet
    from keras_retinanet.losses import smooth_l1, focal
    from keras_retinanet.layers import RegressBoxes
    inp = Input([None, None, 3])
    model, anchors = resnet_retinanet(2, backbone, bbox=False, inputs=inp)
    regression, classification = model.outputs
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    detections = Lambda(non_max_suppression(), name='detections', arguments=nms_args)([boxes, classification])
    model = Model(inp, [regression, classification, detections], name='RetinaNet')

    loss = dict(regression=smooth_l1(), classification=focal())
    optimizer = Adam(lr=1e-5, clipnorm=0.001)
    model.compile(loss=loss, optimizer=optimizer)
    if weights: model.load_weights(weights)
    return model

def load_model(weights, nms_args={}, compile=False):
    from keras.models import model_from_yaml
    from keras.optimizers import Adam
    from keras_retinanet.losses import smooth_l1, focal
    from keras_retinanet.models.resnet import custom_objects
    model_dir = os.path.dirname(weights)
    config_dir = model_dir
    config = os.path.join(config_dir, 'config.yml')
    while not os.path.exists(config):
        config_dir = os.path.dirname(config_dir)
        config = os.path.join(config_dir, 'config.yml')
        if os.path.relpath(config, 'models/config.yml') == '.':
            raise FileNotFoundError('Model config not found.')
    print('Load model config %s' % config)

    with open(config, 'r') as f:
        config = f.read()
    custom_objects['non_max_suppression'] = non_max_suppression(nms_args)
    model = model_from_yaml(config, custom_objects=custom_objects)
    if compile:
        loss = dict(regression=smooth_l1(), classification=focal())
        optimizer = Adam(lr=1e-5, clipnorm=0.001)
        model.compile(loss=loss, optimizer=optimizer)
    model.load_weights(weights)
    return model

def model_name(name):
    for dir in ['logs', 'models']:
        if name.startswith(dir): 
            name = os.path.relpath(name, dir)
            if os.path.isfile(name):
                name = os.path.dirname(name)
    return name

def model_path(name=None):
    name = model_name(name)
    dirs = []
    for dir in ['logs', 'models']:
        dir = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        dirs.append(dir)

    return dirs

def delete_model(name, dry=False):
    from shutil import rmtree
    err = []
    for path in model_path(name):
        print('delete', path)
        if not dry:
            rmtree(path, onerror=lambda *args: err.append(args))
    return err

def find_best(name='**'):
    import numpy as np
    import re
    name = model_name(name)
    weights = glob('models/%s/*.h5' % name, recursive=True)
    ex = re.compile(u'/weights\.[0-9]*-(.*)\.h5')
    scores = [float(ex.search(weight).groups()[0]) for weight in weights]
    max_id = np.argmax(scores)
    return weights[max_id], scores[max_id]

def clear_empty(name='**', dry=False):
    for model_dir in glob('models/%s/' % name, recursive=True):
        if not glob(model_dir + '/**/weights.*-*.h5', recursive=True):
            delete_model(model_dir, dry)
