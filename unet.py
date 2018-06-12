import numpy as np
import pandas as pd
from skimage import filters
from scipy.ndimage.filters import uniform_filter, gaussian_filter
from skimage.io import imsave, imshow
from keras.models import Model
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.utils import plot_model
from keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.morphology import label
from keras import backend as K
from config import FILE_PATH, RANDOM_STATE
from config import IMG_CHAN, IMG_WIDTH, IMG_HEIGHT, MERG_RATION
from config import BATCH_SIZE, EPOCHS, NU, MU

from keras.preprocessing.image import ImageDataGenerator


def generator(x_train, y_train, batch_size):
    data_gen_args = dict(shear_range=0.5,
                         rotation_range=50,
                         zoom_range=0.2,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         fill_mode='reflect'

    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(x_train, seed=RANDOM_STATE)
    mask_datagen.fit(y_train, seed=RANDOM_STATE)
    image_generator = image_datagen.flow(x_train,
                                         batch_size=batch_size,
                                         seed=RANDOM_STATE)
    mask_generator = mask_datagen.flow(y_train,
                                       batch_size=batch_size,
                                       seed=RANDOM_STATE)
    train_generator = zip(image_generator, mask_generator)
    return train_generator



def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2 * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    loss = []
    for t in np.arange(3, 102, 3):
        y_pred_ = (t-2.5-y_pred)/(t-0.5)
        y_true_ = (t-2.5-y_true)/(t-0.5)
        loss.append(dice_coef_loss(y_true_, y_pred_))
    return K.mean(K.stack(loss), axis=0)


# Define IoU metric
def mean_iou(y_true, y_pred):
   score, up_opt = K.tf.metrics.mean_iou(y_true, y_pred, 2)
   K.get_session().run(K.tf.local_variables_initializer())
   with K.tf.control_dependencies([up_opt]):
       score = K.tf.identity(score)
   return score



def conv_layer(feature_batch, feature_map, activation='lrelu', kernel_size=(3, 3), strides=(1, 1), dp_ration=0.):
    conv = Conv2D(filters=feature_map,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_regularizer=regularizers.l2(0.),
                  kernel_initializer='he_normal',
                  padding='same')(feature_batch)
    bn = BatchNormalization(axis=3)(conv)
    if activation == 'lrelu':
        act = LeakyReLU(1/3)(bn)
    else:
        act = Activation(activation=activation)(bn)
    dp = Dropout(dp_ration)(act)
    return dp


def conv_transpose_layer(feature_batch, feature_map, kernel_size=(2, 2), strides=(2, 2)):
    dconv = Conv2DTranspose(filters=feature_map,
                            kernel_size=kernel_size,
                            strides=strides,
                            kernel_regularizer=regularizers.l2(0.),
                            padding='same')(feature_batch[0])
    con = concatenate([dconv, feature_batch[1]])
    return con


def uconv_block(feature_batch, feature_map, activation='lrelu', dp_ration=0.):
    conv1 = conv_layer(feature_batch,
                       feature_map,
                       activation=activation,
                       dp_ration=dp_ration)
    conv2 = conv_layer(conv1,
                       feature_map,
                       activation=activation)
    mp = MaxPooling2D((2, 2))(conv2)
    return mp, conv2


def uchain_block(feature_batch, feature_map, activation='lrelu', dp_ration=0.):
    conv1 = conv_layer(feature_batch,
                       feature_map,
                       activation=activation,
                       dp_ration=dp_ration)
    conv2 = conv_layer(conv1,
                       feature_map,
                       activation=activation)
    return conv2


def udconv_block(feature_batch, feature_map, activation='lrelu', dp_ration=0.):
    dconv = conv_transpose_layer(feature_batch,
                                 feature_map)
    conv1 = conv_layer(dconv,
                       feature_map,
                       activation=activation,
                       dp_ration=dp_ration)
    conv2 = conv_layer(conv1,
                       feature_map,
                       activation=activation)
    return conv2


def get_callbacks(filepath, patience=10):
    lr_reduce = ReduceLROnPlateau(monitor='loss',
                                  factor=0.1,
                                  epsilon=1e-4,
                                  mode='max',
                                  patience=patience,
                                  min_lr=1e-5,
                                  verbose=1)
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [lr_reduce, msave]


def get_model():
    inp = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHAN))
    norm = Lambda(lambda x: x/255)(inp)

    uconv1 = uconv_block(norm, 16, dp_ration=0., activation='elu')
    uconv2 = uconv_block(uconv1[0], 32, dp_ration=0.1, activation='elu')
    uconv3 = uconv_block(uconv2[0], 64, dp_ration=0.1, activation='elu')
    uconv4 = uconv_block(uconv3[0], 128, dp_ration=0.1, activation='elu')

    uchain = uchain_block(uconv4[0], 256, dp_ration=0.1, activation='elu')

    udconv1 = udconv_block([uchain, uconv4[1]], 128, dp_ration=0.1, activation='elu')
    udconv2 = udconv_block([udconv1, uconv3[1]], 64, dp_ration=0.1, activation='elu')
    udconv3 = udconv_block([udconv2, uconv2[1]], 32, dp_ration=0.1, activation='elu')
    udconv4 = udconv_block([udconv3, uconv1[1]], 16, dp_ration=0., activation='elu')

    out = conv_layer(udconv4, 2, activation='sigmoid', kernel_size=(1, 1))

    #conv1 = conv_layer(out_conv, 16, activation='elu')
    #add1 = add([out_conv, conv1])
    #conv2 = conv_layer(conv1, 16, activation='elu')
    #add2 = add([add1, conv2])
    #conv3 = conv_layer(conv2, 16, activation='elu')
    #add3 = add([add2, conv3])
    #conv4 = conv_layer(conv3, 16, activation='elu')
    #add4 = add([add3, conv4])

    #out = conv_layer(add4, 1, activation='relu', kernel_size=(1, 1))

    model = Model(inputs=[inp], outputs=[out])

    opt = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #opt = SGD(lr=0.5, momentum=0.9, nesterov=True, decay=0.1)
    model.compile(optimizer=opt,
                  loss=dice_coef_loss,
                  metrics=[mean_iou])

    model.summary()
    """plot_model(model=model,
               to_file='unet.png',
               show_layer_names=True,
               show_shapes=True)"""
    return model


def train_model(X, y, name=''):
    gmodel = get_model()
    X = np.array(X[int(len(X)*MU):int(len(X)*NU)])
    y = np.array(y[int(len(y)*MU):int(len(y)*NU)])
    #gmodel.load_weights(FILE_PATH+'model_weights'+name+'.hdf5')
    callbacks = get_callbacks(filepath=FILE_PATH+'model_weights'+name+'.hdf5',
                              patience=5)

    gmodel.fit(X, y, batch_size=BATCH_SIZE,
               epochs=EPOCHS,
               verbose=1,
               shuffle=True,
               validation_split=0.2,
               callbacks=callbacks)


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def depader(img, shape):
    img = img[0:shape[0], :shape[1], :]
    return img

def concater(imgs, img_shape, imgs_pos, t_flag=False):
    if len(img_shape)==2:
        img_shape = [img_shape[0], img_shape[1], 2]
    img = []
    for pimg, (i, j) in zip(imgs, imgs_pos):
        sx1 = int(IMG_WIDTH*(i/MERG_RATION))
        sx2 = int(IMG_WIDTH*(i/MERG_RATION+1))
        sy1 = int(IMG_HEIGHT*(j/MERG_RATION))
        sy2 = int(IMG_HEIGHT*(j/MERG_RATION+1))
        canva = np.full(img_shape, np.nan)
        if t_flag:
            sx2, sx1 = img_shape[0] - sx1, img_shape[0] - sx2
            sy2, sy1 = img_shape[1] - sy1, img_shape[1] - sy2

        canva[sx1:sx2, sy1:sy2, :] = pimg[:, :, :]
        img.append(canva)
    img = np.array(img)
    img = np.nanmedian(img, axis=0)
    img[img==np.nan] = 0
    return np.reshape(img, img_shape)


def meanum(img1, img2, img3):
    img = np.nanmin([img1, img2], axis=0)
    img = np.nanmin([img, img3], axis=0)
    return img

def maximum(img1, img2, img3):
    img = np.maximum(img1, img2)
    img = np.maximum(img, img3)
    return img

def hor_flip(img):
    return img[::-1,:,:]

def vert_flip(img):
    return img[:,::-1,:]

def part2all(img_dict):
    imgs = img_dict['img']
    #vimgs = img_dict['vimg']
    #himgs = img_dict['himg']
    #timgs = img_dict['timg']
    #tvimgs = img_dict['vtimg']
    #thimgs = img_dict['htimg']
    img_shape = img_dict['shape']
    pos = img_dict['pos']

    x, y, c = img_shape
    x = x + IMG_WIDTH
    y = y + IMG_HEIGHT
    sx = x - x//IMG_WIDTH
    sy = y - y//IMG_HEIGHT

    img = concater(imgs, [x, y], pos)
    #vimg = concater(vimgs, [x, y], pos)
    #himg = concater(himgs, [x, y], pos)

    #timg = concater(timgs, [x, y], pos, t_flag=True)
    #tvimg = concater(tvimgs, [x, y], pos, t_flag=True)
    #thimg = concater(thimgs, [x, y], pos, t_flag=True)

    #vimg = vert_flip(vimg)
    #himg = hor_flip(himg)
    #tvimg = vert_flip(tvimg)
    #thimg = hor_flip(thimg)

    #tres_img = meanum(img[:, :, :], vimg[:, :, :], himg[:, :, :])

    #res_img = np.mean([res_img, tres_img], axis=0)
    dep_img = depader(img, shape=img_shape)
    #dep_vimg = depader(vimg, shape=img_shape)
    #dep_himg = depader(himg, shape=img_shape)
    #res_img = meanum(dep_img[:, :, :], dep_vimg[:, :, :], dep_himg[:, :, :])
    #return np.reshape(dep_img, dep_img.shape[:2])
    return dep_img

def predict_set(img_set, model):
    pred_dict = {}
    gmodel = model
    for l in img_set:
        if 'img' in l:
            x = np.array(img_set[l])
            pred = gmodel.predict(x, verbose=1)
            pred_dict[l] = pred
        else:
            pred_dict[l] = img_set[l]
    del gmodel
    return pred_dict


def test_model(imgs_set, test_id, cluster,name=''):
    new_test_ids = []
    rles = []
    m = []

    for i in [0,3]:
        gmodel = get_model()
        gmodel.load_weights(filepath=FILE_PATH+'model_weights'+str(i)+'.hdf5')
        m.append(gmodel)

    for i,img_set in enumerate(imgs_set):
        model = m[int(int(cluster[i])!=0)]
        p_set = predict_set(img_set, model=model)
        p = part2all(p_set)
        #p = filters.median(p)
        p1 = p [:,:,0]
        p2 = p [:,:,1]
        p = p1-p2
        #p = filters.threshold_niblack(p, window_size=3, k=0.1)
        val = filters.threshold_otsu(p)
        p = (p>val).astype(np.uint8)
        imsave(FILE_PATH+'pred/'+str(cluster[i])+'/'+str(test_id[i])+'.png', p*255)
        rle = list(prob_to_rles(p))
        rles.extend(rle)
        new_test_ids.extend([test_id[i]] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub-dsbowl2018_2_otsu_nanminmin.csv', index=False)
