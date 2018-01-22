import os
import re
import json
import numpy as np

def get_name(name=None, prefix='retinanet'):
    from time import strftime
    return name or '%s/%s.%s' % (
            prefix, strftime('%y%m%d-%H%M%S'), os.uname().nodename)
def get_session():
    import tensorflow as tf

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def record_hyperparameters(args, writer):
    import tensorflow as tf
    import yaml
    args = np.string_([[k, str(v)] for k, v in vars(args).items()])
    with get_session().as_default() as sess:
        args = tf.get_variable('args', dtype=tf.string, initializer=args)
        summ_op = tf.summary.text('args', args.value())
        sess.run(args.initializer)
        summ = sess.run(summ_op)
    writer.add_summary(summ, 0)
    return args

def set_tf_environ(visible_devices, log_level):
    if visible_devices: 
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level
