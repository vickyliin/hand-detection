
# coding: utf-8

# syn data
# /nfs/nas-5.1/cwtsai/ADLxMLDSFinal/DeepQ-Synth-Hand-01/data
# both_hand_n : 15839
# left_hand_only_n : 17075
# right_hand_only_n : 17086

# /nfs/nas-5.1/cwtsai/ADLxMLDSFinal/DeepQ-Synth-Hand-02/data
# both_hand_n : 15820
# left_hand_only_n : 17198
# right_hand_only_n : 16982

# total data:
# both_hand_n : 31659
# left_hand_only_n : 34273
# right_hand_only_n : 34068

# vive data
# /nfs/nas-5.1/cwtsai/ADLxMLDSFinal/DeepQ-Vivepaper/data
# both_hand_n : 222
# left_hand_only_n : 191
# right_hand_only_n : 51

#Initialization
import os
import sys
import numpy as np
import pandas as pd
import random
from collections import deque
from sklearn.utils import shuffle
import tensorflow as tf
import keras
from keras.applications import resnet50, vgg16
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
import h5py
import cv2

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

def init():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)
init()

do_pp = True
SYN_ONLY = False
REAL_ONLY = False
half_data = True
filter_syn = False
use_net = 'resnet' # 'resnet' or 'vgg16' or None
see_result = False
load_model = True

# h = 460
# w = 612
h = 240
w = 320

data_path_syn1 = '/nfs/nas-5.1/cwtsai/ADLxMLDSFinal/DeepQ-Synth-Hand-01/data'
data_path_syn2 = '/nfs/nas-5.1/cwtsai/ADLxMLDSFinal/DeepQ-Synth-Hand-02/data'
data_path_vive = '/nfs/nas-5.1/cwtsai/ADLxMLDSFinal/DeepQ-Vivepaper/data'
if SYN_ONLY :
    lst_data_path = [data_path_syn1,data_path_syn2]
elif REAL_ONLY :
    lst_data_path = [data_path_vive]
else :
    lst_data_path = [data_path_syn1,data_path_syn2,data_path_vive]

path_temp = ''
if REAL_ONLY :
    path_temp += 'real_only'
if half_data :
    path_temp += 'half'
if filter_syn :
    path_temp += 'filter_syn'



#Preprocessing
def image_pp(data_path,r) :
    im = cv2.resize(cv2.imread('{}/{}'.format(data_path,r['png_path'])), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = np.expand_dims(im, axis=0)
    return im

def pp() :
    # preprocessing
    # input X : image with any size
    #       y :
    # output X : shape=(224,224,3)
    #        y : shape=(4,) which is (x1,y1,x2,y2) note : (x1,y1) is bbox top-left coordinate, (x2,y2) is bbox bottom-right coordinate,
    both_hand_n = 0
    left_hand_only_n = 0
    right_hand_only_n = 0

    df_B = pd.DataFrame(columns=['png_path','x0','y0','x1','y1','LR'] )
    df_L = pd.DataFrame(columns=['png_path','x0','y0','x1','y1','LR'] )
    df_R = pd.DataFrame(columns=['png_path','x0','y0','x1','y1','LR'] )
    for data_path in lst_data_path :
        if filter_syn :
            df_ann = pd.read_csv('{}/filtered_annotations.csv'.format(data_path), names=['png_path','x0','y0','x1','y1','LR'] )
        else :
            df_ann = pd.read_csv('{}/annotations.csv'.format(data_path), names=['png_path','x0','y0','x1','y1','LR'] )
        df_ann['class'] = 0
        if half_data :
            df_ann = df_ann.iloc[:int(df_ann.shape[0]/2)]
        df_ann_sort = df_ann.sort_values(['png_path'])
        df_ann_sort = df_ann_sort.reset_index(drop=True)

        df_ann_sort['x0'] = df_ann_sort['x0']*224/w
        df_ann_sort['y0'] = df_ann_sort['y0']*224/h
        df_ann_sort['x1'] = df_ann_sort['x1']*224/w
        df_ann_sort['y1'] = df_ann_sort['y1']*224/h

        ### create list of three kinds of hands
        lst_both_hand = []
        lst_left_hand_only = []
        lst_right_hand_only = []
        name = 'null'
        lst_class = []
        for i,r in df_ann_sort.iterrows() :
            sys.stdout.write('\r {} \t'.format(i))
            sys.stdout.flush()
            if r['LR'] == 'L' :
                if r['png_path'] == name :
                    # both hand
                    lst_right_hand_only.pop()
                    lst_both_hand += [r['png_path']]
                    lst_class.pop()
                    lst_class += [0,0]
                else :
                    lst_left_hand_only += [r['png_path']]
                    lst_class += [1]
            elif r['LR'] == 'R' :
                if r['png_path'] == name :
                    # both hand
                    lst_left_hand_only.pop()
                    lst_both_hand += [r['png_path']]
                    lst_class.pop()
                    lst_class += [0,0]
                else :
                    lst_right_hand_only += [r['png_path']]
                    lst_class += [2]
            name = r['png_path']
        both_hand_n += len(lst_both_hand)
        left_hand_only_n += len(lst_left_hand_only)
        right_hand_only_n += len(lst_right_hand_only)

        print ('\n{}'.format(data_path))
        print ('both_hand_n : {}'.format(len(lst_both_hand)))
        print ('left_hand_only_n : {}'.format(len(lst_left_hand_only)))
        print ('right_hand_only_n : {}'.format(len(lst_right_hand_only)))

        ### data for C
        print ('C')
        lst_X = []
        lst_y = []
        FLAG_two_hands = 0
        for i,r in df_ann_sort.iterrows() :
            sys.stdout.write('\r {} \t'.format(i))
            sys.stdout.flush()
            if FLAG_two_hands :
                FLAG_two_hands = 0
                continue
            elif lst_class[i] == 0:
                FLAG_two_hands = 1
            im = image_pp(data_path,r)
            lst_X += [im]
            lst_y += [lst_class[i]]

        ary_X = np.vstack(lst_X)
        ary_y = np.asarray(lst_y)
        assert len(ary_X)==len(ary_y), error

        if not os.path.isdir('./data_pp') :
            os.makedirs('./data_pp')
        np.save('./data_pp/ary_class_X_{}_{}.npy'.format(data_path.split('/')[5],path_temp), ary_X)
        np.save('./data_pp/ary_class_y_{}_{}.npy'.format(data_path.split('/')[5],path_temp), ary_y)

        ### data for L
        print ('L')
        df_L = df_ann_sort.loc[df_ann_sort['LR']=='L'].copy().drop(['LR','class'], axis=1)
        lst_X = []
        lst_y = []
        for i,r in df_L.iterrows() :
            sys.stdout.write('\r {} \t'.format(i))
            sys.stdout.flush()
            im = image_pp(data_path,r)
            lst_X += [im]
            lst_y += [r[['x0','y0','x1','y1']]]

        ary_X = np.vstack(lst_X)
        ary_y = np.asarray(lst_y)
        assert len(ary_X)==len(ary_y), error

        if not os.path.isdir('./data_pp') :
            os.makedirs('./data_pp')
        np.save('./data_pp/ary_L_X_{}_{}.npy'.format(data_path.split('/')[5],path_temp), ary_X)
        np.save('./data_pp/ary_L_y_{}_{}.npy'.format(data_path.split('/')[5],path_temp), ary_y)

        ### data for R
        print ('R')
        df_R = df_ann_sort.loc[df_ann_sort['LR']=='R'].copy().drop(['LR','class'], axis=1)
        lst_X = []
        lst_y = []
        for i,r in df_R.iterrows() :
            sys.stdout.write('\r {} \t'.format(i))
            sys.stdout.flush()
            im = image_pp(data_path,r)
            lst_X += [im]
            lst_y += [r[['x0','y0','x1','y1']]]

        ary_X = np.vstack(lst_X)
        ary_y = np.asarray(lst_y)
        assert len(ary_X)==len(ary_y), error

        if not os.path.isdir('./data_pp') :
            os.makedirs('./data_pp')
        np.save('./data_pp/ary_R_X_{}_{}.npy'.format(data_path.split('/')[5],path_temp), ary_X)
        np.save('./data_pp/ary_R_y_{}_{}.npy'.format(data_path.split('/')[5],path_temp), ary_y)

    print ('\ntotal data: ')
    print('both_hand_n : {}'.format(both_hand_n))
    print('left_hand_only_n : {}'.format(left_hand_only_n))
    print('right_hand_only_n : {}'.format(right_hand_only_n))

# Model
if use_net == 'resnet' :
    pre_net = resnet50.ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000)
elif use_net == 'vgg16' :
    pre_net = vgg16.VGG16(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000)

elif use_net == None :
    i = Input(shape=(224,224,3))
    x = Conv2D(32,(3,3),strides=(2,2),activation='selu',kernel_initializer='lecun_normal')(i)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = BatchNormalization()(x)
    x = Conv2D(32,(3,3),strides=(2,2),activation='selu',kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = BatchNormalization()(x)
    x = Conv2D(32,(3,3),strides=(2,2),activation='selu',kernel_initializer='lecun_normal')(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = BatchNormalization()(x)
    x = Conv2D(32,(3,3),strides=(2,2),activation='selu',kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    pre_net = Model(i,x)


if use_net != None :
    pre_net.layers.pop()
    pre_net.layers.pop()
    pre_net.layers.pop()
    pre_net.layers.pop()
    pre_net.layers.pop()
    pre_net.layers.pop()
    for layer in pre_net.layers :
        layer.trainable=False

ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
TB = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
MC_C = keras.callbacks.ModelCheckpoint('./keras_models/model_C.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
MC_L = keras.callbacks.ModelCheckpoint('./keras_models/model_L.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
MC_R = keras.callbacks.ModelCheckpoint('./keras_models/model_R.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

def build_model_C() :
    print (pre_net.summary())
#     x = Dense(256,activation='selu',kernel_initializer='lecun_normal')(pre_net.output)
#     x = BatchNormalization()(x)
#     x = Dense(64,activation='selu',kernel_initializer='lecun_normal')(x)
    x = Conv2D(32,(3,3),strides=(2,2),activation='selu',kernel_initializer='lecun_normal')(pre_net.output)
    x = Conv2D(32,(3,3),strides=(2,2),activation='selu',kernel_initializer='lecun_normal')(x)
    x = Conv2D(32,(3,3),strides=(2,2),activation='selu',kernel_initializer='lecun_normal')(x)
    x = Conv2D(32,(3,3),strides=(2,2),activation='selu',kernel_initializer='lecun_normal')(x)
#     x = Dense(3,activation='softmax')(pre_net.output)
    x = Dense(3,activation='softmax')(x)
    model = Model(pre_net.input,x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

def build_model_L() :
#     x = Dense(256,activation='selu',kernel_initializer='lecun_normal')(pre_net.output)
#     x = BatchNormalization()(x)
#     x = Dense(64,activation='selu',kernel_initializer='lecun_normal')(x)
    x = Dense(4,activation=None)(pre_net.output)
#     x = Dense(4,activation=None)(x)
    model = Model(pre_net.input,x)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_model_R() :
#     x = Dense(256,activation='selu',kernel_initializer='lecun_normal')(pre_net.output)
#     x = BatchNormalization()(x)
#     x = Dense(64,activation='selu',kernel_initializer='lecun_normal')(x)
    x = Dense(4,activation=None)(pre_net.output)
#     x = Dense(4,activation=None)(x)
    model = Model(pre_net.input,x)
    model.compile(optimizer='adam', loss='mse')
    return model

def C_training():
    model = build_model_C()
    model.fit(ary_X_train_C,ary_y_train_C, epochs=200, batch_size=32, validation_split=0.3, callbacks=[ES,TB,MC_C])
    return model

def L_training() :
    model = build_model_L()
    model.fit(ary_X_train_L,ary_y_train_L, epochs=200, batch_size=32, validation_split=0.3, callbacks=[ES,TB,MC_L])
    return model

def R_training() :
    model = build_model_R()
    model.fit(ary_X_train_R,ary_y_train_R, epochs=200, batch_size=32, validation_split=0.3, callbacks=[ES,TB,MC_R])
    return model

# Main
if __name__ == '__main__' :
    if do_pp :
        pp()

    ary_X_vive_C = np.load('./data_pp/ary_class_X_{}_real_only.npy'.format(data_path_vive.split('/')[5]))
    ary_y_vive_C = np.load('./data_pp/ary_class_y_{}_real_only.npy'.format(data_path_vive.split('/')[5]))
    dummy_y_vive_C = np_utils.to_categorical(ary_y_vive_C)
    ary_X_vive_C, ary_y_vive_C = shuffle(ary_X_vive_C, dummy_y_vive_C, random_state=0)
    ary_X_syn1_C = np.load('./data_pp/ary_class_X_{}_{}.npy'.format(data_path_syn1.split('/')[5],path_temp))
    ary_y_syn1_C = np.load('./data_pp/ary_class_y_{}_{}.npy'.format(data_path_syn1.split('/')[5],path_temp))
    dummy_y_syn1_C = np_utils.to_categorical(ary_y_syn1_C)
    ary_X_syn1_C, ary_y_syn1_C = shuffle(ary_X_syn1_C, dummy_y_syn1_C, random_state=0)
    if not half_data :
        ary_X_syn2_C = np.load('./data_pp/ary_class_X_{}_{}.npy'.format(data_path_syn2.split('/')[5],path_temp))
        ary_y_syn2_C = np.load('./data_pp/ary_class_y_{}_{}.npy'.format(data_path_syn2.split('/')[5],path_temp))
        dummy_y_syn2_C = np_utils.to_categorical(ary_y_syn2_C)
        ary_X_syn2_C, ary_y_syn2_C = shuffle(ary_X_syn2_C, dummy_y_syn2_C, random_state=0)

        ary_X_train_C = np.concatenate((ary_X_syn1_C,ary_X_syn2_C,ary_X_vive_C), axis=0)
        ary_y_train_C = np.concatenate((ary_y_syn1_C,ary_y_syn2_C,ary_y_vive_C), axis=0)
    else :
        ary_X_train_C = np.concatenate((ary_X_syn1_C,ary_X_vive_C), axis=0)
        ary_y_train_C = np.concatenate((ary_y_syn1_C,ary_y_vive_C), axis=0)

    model_C = C_training()

    ary_X_vive_L = np.load('./data_pp/ary_L_X_{}_real_only.npy'.format(data_path_vive.split('/')[5]))
    ary_y_vive_L = np.load('./data_pp/ary_L_y_{}_real_only.npy'.format(data_path_vive.split('/')[5]))
    ary_X_vive_L, ary_y_vive_L = shuffle(ary_X_vive_L, ary_y_vive_L, random_state=0)
    ary_X_syn1_L = np.load('./data_pp/ary_L_X_{}_{}.npy'.format(data_path_syn1.split('/')[5],path_temp))
    ary_y_syn1_L = np.load('./data_pp/ary_L_y_{}_{}.npy'.format(data_path_syn1.split('/')[5],path_temp))
    ary_X_syn1_L, ary_y_syn1_L = shuffle(ary_X_syn1_L, ary_y_syn1_L, random_state=0)
    if not half_data :
        ary_X_syn2_L = np.load('./data_pp/ary_L_X_{}_{}.npy'.format(data_path_syn2.split('/')[5],path_temp))
        ary_y_syn2_L = np.load('./data_pp/ary_L_y_{}_{}.npy'.format(data_path_syn2.split('/')[5],path_temp))
        ary_X_syn2_L, ary_y_syn2_L = shuffle(ary_X_syn2_L, ary_y_syn2_L, random_state=0)

        ary_X_train_L = np.concatenate((ary_X_syn1_L,ary_X_syn2_L,ary_X_vive_L), axis=0)
        ary_y_train_L = np.concatenate((ary_y_syn1_L,ary_y_syn2_L,ary_y_vive_L), axis=0)
    else :
        ary_X_train_L = np.concatenate((ary_X_syn1_L,ary_X_vive_L), axis=0)
        ary_y_train_L = np.concatenate((ary_y_syn1_L,ary_y_vive_L), axis=0)
    ary_X_train_L = ary_X_vive_L[:-20].copy()
    ary_y_train_L = ary_y_vive_L[:-20].copy()
    ary_X_test_L = ary_X_vive_L[-20:].copy()
    ary_y_test_L = ary_y_vive_L[-20:].copy()

    model_L = L_training()

    ary_X_vive_R = np.load('./data_pp/ary_L_X_{}_real_only.npy'.format(data_path_vive.split('/')[5]))
    ary_y_vive_R = np.load('./data_pp/ary_L_y_{}_real_only.npy'.format(data_path_vive.split('/')[5]))
    ary_X_vive_R, ary_y_vive_R = shuffle(ary_X_vive_R, ary_y_vive_R, random_state=0)
    ary_X_syn1_R = np.load('./data_pp/ary_L_X_{}_{}.npy'.format(data_path_syn1.split('/')[5],path_temp))
    ary_y_syn1_R = np.load('./data_pp/ary_L_y_{}_{}.npy'.format(data_path_syn1.split('/')[5],path_temp))
    ary_X_syn1_R, ary_y_syn1_R = shuffle(ary_X_syn1_R, ary_y_syn1_R, random_state=0)
    if not half_data :
        ary_X_syn2_R = np.load('./data_pp/ary_L_X_{}_{}.npy'.format(data_path_syn2.split('/')[5],path_temp))
        ary_y_syn2_R = np.load('./data_pp/ary_L_y_{}_{}.npy'.format(data_path_syn2.split('/')[5],path_temp))
        ary_X_syn2_R, ary_y_syn2_R = shuffle(ary_X_syn2_R, ary_y_syn2_R, random_state=0)

        ary_X_train_R = np.concatenate((ary_X_syn1_R,ary_X_syn2_R,ary_X_vive_R), axis=0)
        ary_y_train_R = np.concatenate((ary_y_syn1_R,ary_y_syn2_R,ary_y_vive_R), axis=0)
    else :
        ary_X_train_R = np.concatenate((ary_X_syn1_R,ary_X_vive_R), axis=0)
        ary_y_train_R = np.concatenate((ary_y_syn1_R,ary_y_vive_R), axis=0)

    model_R = R_training()

    # show prediction
    if see_result :
        import glob
        import json
        import numpy as np
        from skimage import img_as_float
        from skimage.draw import line,circle
        from skimage.io import imread, imsave
        import os
        np.random.seed(0)

        def draw_rectangle(img, x0, y0, x1, y1, color=(0, 0, 0)):
            draw_line(img, x0, y0, x0, y1, color)
            draw_line(img, x0, y0, x1, y0, color)
            draw_line(img, x1, y0, x1, y1, color)
            draw_line(img, x0, y1, x1, y1, color)

        def draw_line(img, x0, y0, x1, y1, color=(0, 0, 0)):
            x0 = min(img.shape[1]-1, max(0, x0))
            y0 = min(img.shape[0]-1, max(0, y0))
            x1 = min(img.shape[1]-1, max(0, x1))
            y1 = min(img.shape[0]-1, max(0, y1))
            yy, xx = line(y0,x0,y1,x1)
            img[yy, xx, :] = color

        def draw_bbox(img_path, label_path, output_path):
            img = img_as_float(imread(img_path))
            json_item = json.load(open(label_path))
            colors = {'L':(1,0,0),'R':(0,1,0)}
            for htype,box in json_item['bbox'].items():
                draw_rectangle(img, box[0], box[1], box[2], box[3], color=colors[htype])
                draw_rectangle(img, box[0] + 1, box[1] + 1, box[2] + 1, box[3] + 1, color=(0, 0, 0))
            imsave(output_path, img)

        def draw_bbox_new(img, hand_type, Lbox, Rbox):
            img[:,:,0] += 103.939
            img[:,:,1] += 116.779
            img[:,:,2] += 123.68
            img = np.clip(img,0,255)
            img = img.astype('uint8')
            img = img[..., ::-1]
            img = img_as_float(img)
            colors = {'L':(1,0,0),'R':(0,1,0)}
            if hand_type == 0 :
                if all(x<223 for x in Lbox) and all(x>0 for x in Lbox):
                    draw_rectangle(img, Lbox[0], Lbox[1], Lbox[2], Lbox[3], color=colors['L'])
                    draw_rectangle(img, Lbox[0] + 1, Lbox[1] + 1, Lbox[2] + 1, Lbox[3] + 1, color=(0, 0, 0))
                if all(x<223 for x in Rbox) and all(x>0 for x in Rbox):
                    draw_rectangle(img, Rbox[0], Rbox[1], Rbox[2], Rbox[3], color=colors['R'])
                    draw_rectangle(img, Rbox[0] + 1, Rbox[1] + 1, Rbox[2] + 1, Rbox[3] + 1, color=(0, 0, 0))
            elif hand_type == 1 :
                if all(x<223 for x in Lbox) and all(x>0 for x in Lbox):
                    draw_rectangle(img, Lbox[0], Lbox[1], Lbox[2], Lbox[3], color=colors['L'])
                    draw_rectangle(img, Lbox[0] + 1, Lbox[1] + 1, Lbox[2] + 1, Lbox[3] + 1, color=(0, 0, 0))
            elif hand_type == 2 :
                if all(x<223 for x in Rbox) and all(x>0 for x in Rbox):
                    draw_rectangle(img, Rbox[0], Rbox[1], Rbox[2], Rbox[3], color=colors['R'])
                    draw_rectangle(img, Rbox[0] + 1, Rbox[1] + 1, Rbox[2] + 1, Rbox[3] + 1, color=(0, 0, 0))
            return img

        aryx = np.load('./data_pp/ary_class_X_{}_{}.npy'.format(data_path_vive.split('/')[5],'real_only'))
        aryy = np.load('./data_pp/ary_class_y_{}_{}.npy'.format(data_path_vive.split('/')[5],'real_only'))
        aryy = np_utils.to_categorical(aryy)
        aryx, aryy = shuffle(aryx, aryy, random_state=0)

        from matplotlib import pyplot as plt
        get_ipython().run_line_magic('matplotlib', 'inline')
        def pred(img) :
            ary_img = img.reshape((1,224,224,3))
            hand_type = np.argmax(model_C.predict(ary_img).reshape((3,)))
            Lbox = model_L.predict(ary_img).reshape((4,)).astype(int)
            Rbox = model_R.predict(ary_img).reshape((4,)).astype(int)
            img = draw_bbox_new(img, hand_type, Lbox, Rbox)

            print (hand_type)
            print (Lbox)
            print (Rbox)
            return img
        for i in range(10) :
            img = pred(aryx[i].copy())
            plt.imshow(img)
            plt.show()
