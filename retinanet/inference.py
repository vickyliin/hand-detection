import pdb
import sys
import numpy as np
import os.path

def get_best_box(detection):
    coord, score, label = np.hsplit(detection, [4, 5])
    coord = np.int32(coord)
    label, score = label[:, 0], score[:, 0]
    boxes = {}
    for l in range(2):
        idx = np.where(label == l)[0]
        if len(idx) == 0: continue
        max_idx = idx[score[idx].argmax()]
        best_coord = coord[max_idx]
        if all(best_coord == 0): continue
        boxes[l] = list(best_coord), score[max_idx]
    return boxes

def parse_args(args=sys.argv[1:]):
    from arguments import (ParserGroup, ArgumentParser, 
                           GeneratorArgparser,
                           NMSArgparser,
                           TFArgparser,
                           InferenceArgparer)
    return ParserGroup(
        TFArgparser(),
        GeneratorArgparser(), 
        NMSArgparser(set_defaults=False),
        InferenceArgparer(), 
        name='inference'
    ).parse(args)

def make_generator(imgs, **kwargs):
    from keras_retinanet.preprocessing import InputGenerator
    return InputGenerator(imgs, **kwargs)

def inference(model, input_generator, f=sys.stdout, sep=',', base_dir=None):
    from tqdm import tqdm
    pbar = tqdm(total=input_generator.size())
    for imgs in input_generator:
        detections = model.predict_on_batch(imgs.input)[-1]
        for path, scale, detection in zip(imgs.path, imgs.scale, detections):
            if base_dir: path = os.path.relpath(path, base_dir)
            for *c, p, l in detection:
                if not p: continue
                s = sep.join([path, *('%d' % (ci/scale) for ci in c), '%d'%l, '%f'%p])
                pbar.write(s)
                s += '\n'
                f.write(s.encode('ascii'))
            pbar.update(1)
    pbar.close()

def main(args=sys.argv[1:]):
    tf_args, generator_args, nms_args, inference_args = parse_args(args)[1:]
    from utils import set_tf_environ
    set_tf_environ(**vars(tf_args))

    import judger_hand
    from model import load_model
    imgs = inference_args.inputs or judger_hand.get_file_names()
    f = inference_args.output or judger_hand.get_output_file_object()
    input_generator = make_generator(imgs, **vars(generator_args))
    model = load_model(inference_args.weights, vars(nms_args))
    sep=',' if inference_args.output else ' '
    base_dir = os.path.dirname(f.name) if inference_args.output else None
    inference(model, input_generator, f, sep, base_dir)

    if not inference_args.output:
        score, err = judger_hand.judge()
        print('score', score)
        if err is not None:  # in case we failed to judge your submission
            raise Exception(err)
        return score

    else:
        return f

if __name__ == '__main__':
    main()
