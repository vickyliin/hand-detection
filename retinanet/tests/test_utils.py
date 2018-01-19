
def test_record_hyperparameters():
    import tensorflow as tf
    from argparse import Namespace
    from model import delete_model
    from utils import record_hyperparameters
    tf.reset_default_graph()
    delete_model('test')
    writer = tf.summary.FileWriter('logs/test')
    args = Namespace(int=1, float=0.5, bool=True, 
                     list=['a', 'b'], string='hello',
                     dict=dict(x=1.123, y=0.05))
    args = record_hyperparameters(args, writer)
    assert args.shape[0].value == 6
    assert args.shape[1].value == 2

