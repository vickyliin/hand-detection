import os
import sys
def parse_args(args=sys.argv[1:]):
    from arguments import (ParserGroup,
                           TFArgparser,
                           CSVGeneratorArgparser, 
                           ModelArgparser, 
                           TrainArgparser,
                           LRArgparser,
                           NMSArgparser)

    return ParserGroup(
        TFArgparser(),
        CSVGeneratorArgparser(), 
        ModelArgparser(), 
        TrainArgparser(), 
        LRArgparser(), 
        NMSArgparser(), 
        name='train'
    ).parse(args)

def make_generators(data_train, data_valid, **kwargs):
    from keras_retinanet.preprocessing import CSVGenerator
    from keras.preprocessing.image import ImageDataGenerator
    return [CSVGenerator(*data, **kwargs) for data in (data_train, data_valid)]

def train(model,
          train_generator, 
          valid_generator, 
          epochs, 
          lr=None, train_steps=None, valid_steps=None, 
          name=None, writer=None, **kwargs):
    import tensorflow as tf
    from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback
    from keras.optimizers import Adam
    from callbacks import LogMetrics, DrawBBox
    from utils import get_name
    from model import model_path

    name = get_name(name)
    log_dir = model_path(name)[0]
    writer = writer or tf.summary.FileWriter(log_dir)

    callbacks = []
    if lr is not None: 
        callbacks.append(ReduceLROnPlateau(**lr))
    callbacks.append(DrawBBox(writer, train_generator, 'train'))
    callbacks.append(DrawBBox(writer, valid_generator, 'valid'))
    callbacks.append(LogMetrics(name, valid_generator))
    callbacks.append(LambdaCallback(on_epoch_begin=lambda *_: print(name)))

    model.fit_generator(train_generator, 
                        validation_data=valid_generator, 
                        validation_steps=valid_steps or len(valid_generator.groups), 
                        steps_per_epoch=train_steps or len(train_generator.groups), 
                        epochs=epochs, verbose=2, callbacks=callbacks)

def main(args=sys.argv[1:]):
    args, tf_args, generator_args, model_args, train_args, lr_args, nms_args = parse_args(args)
    from utils import set_tf_environ
    set_tf_environ(**vars(tf_args))

    import tensorflow as tf
    import keras.backend as K
    from utils import get_session, get_name, record_hyperparameters
    from model import build_model, model_path
    K.set_session(get_session())

    name = get_name(train_args.__dict__.pop('name'))
    log_dir, model_dir = model_path(name)
    print(name)
    writer = tf.summary.FileWriter(log_dir)
    record_hyperparameters(args, writer)

    train_generator, valid_generator = make_generators(**vars(generator_args))
    model = build_model(**vars(model_args), nms_args=vars(nms_args))
    with open('%s/config.yml' % model_dir, 'w') as f:
        f.write(model.to_yaml())
    try:
        train(model, 
              train_generator, 
              valid_generator, 
              name=name, 
              writer=writer,
              lr=vars(lr_args),
              **vars(train_args))
    except KeyboardInterrupt:
        pass

    return model, name

if __name__ == '__main__':
    main()
