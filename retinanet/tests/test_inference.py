import os
from glob import glob

def test_get_best_box():
    from inference import get_best_box
    import numpy as np
    detection = np.array([
        [1, 2, 3, 4, 0.9, 1],
        [2, 3, 4, 5, 0.8, 1],
        [3, 4, 5, 6, 0.7, 0]
    ])
    boxes = get_best_box(detection)
    assert type(boxes) == dict
    assert len(boxes) == 2
    assert boxes[0] == [3, 4, 5, 6]
    assert boxes[1] == [1, 2, 3, 4]

def test_main():
    from inference import main
    from model import find_best
    weights = find_best()[0]
    print(weights)
    args = ('--weights %s --batch-size 1' % weights).split()
    score = main(args)
    assert score > 0
