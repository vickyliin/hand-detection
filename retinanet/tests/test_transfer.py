
def test_inference_train():
    from glob import glob
    from model import load_model, find_best, delete_model
    import keras.backend as K

    from utils import get_session
    from transfer import inference_train

    name = 'test/transfer'
    delete_model(name)
    K.set_session(get_session())
    imgs_path = glob('data/DeepQ-Vivepaper/frame/**/*.png', recursive=True)[:100]
    valid_csv = 'data/DeepQ-Vivepaper/data/annotations.csv'
    class_map = 'data/class-map.csv'
    thresh = 0.98
    model = load_model(find_best()[0], compile=True)
    buff = inference_train(model, imgs_path, valid_csv, class_map, thresh, 
                           batch_size=4, 
                           image_min_side=460, 
                           image_max_side=1024,
                           name=name)
    buff.seek(0)
    print(buff.read())
