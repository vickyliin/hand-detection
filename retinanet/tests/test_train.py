import os
from glob import glob

def test_main():
    from train import main
    from model import model_path, delete_model
    delete_model('test')
    args = (
        '--train-steps 5 '
        '--epochs 2 '
        '--valid-steps 1 '
        '--batch-size 1 '
        '--name test '
    ).split()
    model, name = main(args)
    model_dir = model_path(name)[1]
    weights = glob(model_dir + '/weights.*-*.h5')
    assert weights, model_dir
    assert os.path.exists('%s/config.yml' % model_dir)
