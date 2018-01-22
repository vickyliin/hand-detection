import numpy as np
from tqdm import tqdm
from judger_hand import _voc_eval

def voc_eval(model, generator):
    splitlines, class_recs = [[], []], [{}, {}] # two classes
    pbar = tqdm(desc='voc eval', total=generator.size())
    for i, (imgs, targets) in enumerate(generator.iterate_once()):
        scales = np.array(imgs.scale)
        detections = model.predict_on_batch(imgs.input)[-1]
        detections[..., :4] /= scales
        for j, bboxes in enumerate(targets.annotations):
            bboxes[..., :4] /= scales
            for k in range(2):
                bboxes_k = bboxes[np.where(bboxes[:, 4] == k)]
                class_recs[k][i,j] = dict(det=[False]*4, bbox=bboxes_k[:, :4])
            for *coords, score, label in detections[j]:
                splitlines[int(label)].append([(i,j), score, *coords])
            pbar.update(1)
    scores = [_voc_eval(s, c) for s, c in zip(splitlines, class_recs)]
    pbar.close()
    return scores

