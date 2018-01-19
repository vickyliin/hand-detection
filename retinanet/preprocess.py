import sys
from glob import glob

def make_annotation_csv(base_dir):
    import os
    import re
    import json
    import numpy as np
    from tqdm import tqdm
    from skimage.io import imread

    base_dir = os.path.join(base_dir, 'data')
    regex = re.compile('%s/(.*)/img/img_(.*).png' % base_dir)
    with open('%s/annotations.csv' % base_dir, 'w') as f:
        filenames = glob(regex.pattern.replace('(.*)', '*'))
        h, w, _ = imread(filenames[0]).shape
        for fn in tqdm(filenames, desc=base_dir):
            id = regex.search(fn).groups()
            with open('%s/%s/label/label_%s.json' % (base_dir, *id), 'r') as ff:
                label = json.load(ff)['bbox']
            for class_name, coords in label.items():
                x0, y0, x1, y1 = coords
                x0, x1 = np.clip([x0, x1], 0, w)
                y0, y1 = np.clip([y0, y1], 0, h)
                if x1 > x0 and y1 > y0:
                    coords = x0, y0, x1, y1
                    fn_rel = os.path.relpath(fn, base_dir)
                    print(fn_rel, *coords, class_name, sep=',', file=f)

if __name__ == '__main__':
    base_dir = sys.argv[1:2] or 'data'
    base_dirs = sys.argv[2:] or glob(base_dir + '/DeepQ-*')

    for base_dir in base_dirs:
        make_annotation_csv(base_dir)
