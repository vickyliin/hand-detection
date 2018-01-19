import pdb
import sys
import numpy as np

def parse_args(args=sys.argv[1:]):
    from arguments import (ArgumentParser,
                           ParserGroup,
                           TFArgparser,
                           CSVGeneratorArgparser, 
                           NMSArgparser,
                           InferenceArgparer)
    parser = ArgumentParser()
    parser.add_argument('--annotate-thresh', type=float, default=0.9)
    parser.add_argument('--thresh-step', type=float, default=0.05)
    parser.add_argument('--name')

    return ParserGroup(
        TFArgparser(),
        CSVGeneratorArgparser(data_train=False), 
        NMSArgparser(), 
        InferenceArgparer(),
        parser,
        name='transfer'
    ).parse(args)

def inference_train(model, 
                    imgs_path, 
                    data_valid, 
                    csv_class_file,
                    annotate_thresh, 
                    thresh_step=0.05,
                    sep=' ',
                    max_epochs=100,
                    n_imgs=10,
                    output=None, 
                    name=None,
                    writer=None,
                    group_method='random',
                    log_steps=100,
                    **generator_args):
    from io import StringIO
    from tqdm import tqdm
    from keras_retinanet.preprocessing import CSVGenerator, InputGenerator
    from utils import get_name
    from callbacks import LogMetrics, DrawBBox
    from inference import get_best_box
    from visualize import draw_bbox

    valid_csv = data_valid
    class_map = csv_class_file
    thresh = annotate_thresh

    name = get_name(name, 'transfer')
    buff = StringIO()
    imgs_path = set(imgs_path)
    valid_generator = CSVGenerator(*valid_csv, 
                                   csv_class_file=class_map, 
                                   group_method=group_method,
                                   **generator_args)

    logger = LogMetrics(name, valid_generator)
    logger.model = model
    drawer = DrawBBox(writer, name=name)
    logger.on_train_begin()

    pbar = tqdm(desc='selected imgs', total=len(imgs_path))
    step = 0

    for epo in range(max_epochs):
        if not imgs_path: break

        # inference
        input_generator = InputGenerator(list(imgs_path), **generator_args)
        pbar_inf = tqdm(desc='inference %03d' % epo, total=input_generator.size())
        bbox_imgs = []

        for batch_imgs in input_generator:

            batch_detections = model.predict_on_batch(batch_imgs.input)[-1]
            for path, img, scale, detections in zip(batch_imgs.path, batch_imgs.resized, batch_imgs.scale, batch_detections):

                boxes = get_best_box(detections)
                if all(p < thresh for _, p in boxes.values()): 
                    pbar_inf.update(1)
                    continue

                for l, (c, p) in boxes.items():

                    if path in imgs_path: 
                        imgs_path.remove(path)
                        pbar.update(1)

                    if len(bbox_imgs) < n_imgs:
                        bbox_img = draw_bbox(img[...,::-1], boxes)
                        bbox_imgs.append(bbox_img)

                    s = sep.join([path, *('%d' % (ci/scale) for ci in c), '%d'%l, '%f'%p])
                    s += '\n'
                    buff.write(s)
                    if output is not None:
                        output.write(s.encode('ascii'))

                pbar_inf.update(1)


        if not bbox_imgs: 
            thresh -= thresh_step
            pbar_inf.close()
            continue
        bbox_imgs_tile = drawer.tile_up(bbox_imgs)
        drawer.add_summary(bbox_imgs_tile, epo)
        pbar_inf.close()


        # train
        buff.seek(0)
        train_generator = CSVGenerator(buff, csv_class_file=class_map, 
                                       sep=sep, base_dirs=['.'], thresh=thresh, 
                                       group_method=group_method, **generator_args)
        pbar_tra = tqdm(desc='train %03d' % epo, total=train_generator.size())
        logs = {name: [] for name in model.metrics_names}

        for batch_imgs, batch_targets in train_generator.iterate_once():

            inputs = batch_imgs.input
            targets = [batch_targets.regression, batch_targets.labels]

            metrics = model.train_on_batch(inputs, targets)

            for name, val in zip(model.metrics_names, metrics):
                logs[name].append(val)

            pbar_tra.update(len(inputs))
            step += 1

            if step % log_steps == 0:
                logs = {name: np.mean(val) for name, val in logs.items()}
                logger.on_epoch_end(step, logs)
                logs = {name: [] for name in model.metrics_names}

        pbar_tra.close()

    pbar.close()
    return buff

def main(args=sys.argv[1:]):
    args, tf_args, generator_args, nms_args, inference_args, transfer_args = parse_args(args)
    from utils import set_tf_environ
    set_tf_environ(**vars(tf_args))

    import tensorflow as tf
    import keras.backend as K
    from utils import get_session, get_name, record_hyperparameters
    from model import build_model, model_path
    K.set_session(get_session())

    import judger_hand
    from model import load_model

    imgs = inference_args.inputs or judger_hand.get_file_names()
    output = inference_args.output or judger_hand.get_output_file_object()
    model = load_model(inference_args.weights, vars(nms_args), compile=True)
    sep = ',' if inference_args.output else ' '

    name = get_name(transfer_args.__dict__.pop('name'), 'transfer')
    log_dir, model_dir = model_path(name)
    print(name)
    writer = tf.summary.FileWriter(log_dir)
    record_hyperparameters(args, writer)

    with open('%s/config.yml' % model_dir, 'w') as f:
        f.write(model.to_yaml())

    try:
        buff = inference_train(model, imgs, output=output, sep=sep,
                               **vars(generator_args),
                               **vars(transfer_args), 
                               name=name,
                               writer=writer)
    except KeyboardInterrupt:
        pass

    if not inference_args.output:
        score, err = judger_hand.judge()
        print('score', score)
        if err is not None:  # in case we failed to judge your submission
            raise Exception(err)
        return score

    return model, name

if __name__ == '__main__':
    main()
