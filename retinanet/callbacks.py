from functools import partial
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback
from inference import get_best_box
from visualize import draw_bbox
from model import model_path

class LogMetrics(Callback):
    def __init__(self, name, generator, valid_prefix='val_'):
        from evaluate import voc_eval
        self.valid_prefix = valid_prefix
        self.log_dir, self.model_dir = model_path(name)
        self.voc_eval = partial(voc_eval, generator=generator)

    def on_train_begin(self, logs={}):
        self.writers = [tf.summary.FileWriter('%s/train' % self.log_dir), 
                        tf.summary.FileWriter('%s/valid' % self.log_dir)]
        self.max_voc = 0
        self.max_voc_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        voc_l, voc_r = self.voc_eval(self.model)
        voc = (voc_l + voc_r) / 2
        logs_ = {}, dict(voc_L=voc_l, voc_R=voc_r, voc=voc) # train, valid
        print(' ', end='')
        for k, v in logs_[1].items():
            print('- '+k, v, sep=': ', end=' ')
        print()
        
        for key, val in logs.items():
            i = 0
            if key.startswith(self.valid_prefix):
                key = key[len(self.valid_prefix):]
                i = 1
            logs_[i][key] = val
        for writer, log in zip(self.writers, logs_):
            summary = [tf.Summary.Value(tag=key, simple_value=val) for key, val in log.items()]
            writer.add_summary(tf.Summary(value=summary), epoch)
            writer.flush()

        if voc >= self.max_voc or epoch >= (self.max_voc_epoch + 100):
            self.max_voc_epoch = epoch
            self.max_voc = voc
            fname = '%s/weights.%03d-%.2f.h5' % (self.model_dir, epoch, voc)
            self.model.save_weights(fname)

    def on_train_end(self, logs={}):
        for writer in self.writers:
            writer.close()

class DrawBBox(Callback):
    def __init__(self, writer=None, generator=None, name=None):
        self.writer = writer = writer or tf.summary.FileWriter(model_path(name)[0])
        self.name = name

        if generator:
            group = generator.groups[0]
            self.imgs = generator.compute_inputs(group)
       
        sess = K.get_session()
        with sess.graph.as_default():
            img_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='bbox_img')
            img_summ = tf.summary.image(name, img_tensor)

        def add_summary(img, step):
            img = sess.run(img_summ, feed_dict={img_tensor: img})
            writer.add_summary(img, step)
            writer.flush()
            
        self.add_summary = add_summary
        
    @staticmethod
    def tile_up(imgs):
        return np.stack(imgs, axis=1).reshape(1, imgs[0].shape[0], -1, imgs[0].shape[-1])

    def on_epoch_end(self, epoch, logs={}):
        detections = self.model.predict_on_batch(self.imgs.input)[-1]
        bbox_imgs = []
        for img, detection in zip(self.imgs.resized, detections):
            boxes = get_best_box(detection)
            bbox_img = draw_bbox(img, boxes)
            bbox_imgs.append(bbox_img)
            print(' -', self.name, end=' - ')
            for l, c in boxes.items():
                print(l, list(c), sep=': ', end=' - ')
            print()
        bbox_imgs_tile = self.tile_up(bbox_imgs)
        self.add_summary(bbox_imgs_tile, epoch)
