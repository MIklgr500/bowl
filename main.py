import os
import sys

import numpy as np
import pandas as pd

from skimage.transform import resize
from skimage.io import imread, imread_collection
from skimage.filters import scharr

from tqdm import tqdm
from random import seed

from unet import train_model, test_model
from config import TRAIN_PATH, TEST_PATH, RANDOM_STATE
from config import IMG_CHAN, IMG_WIDTH, IMG_HEIGHT, MERG_RATION


seed(RANDOM_STATE)

TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % TRAIN_PATH
TRAIN_MASK_PATTERN = "%s/{}/masks/*.png" % TRAIN_PATH
TEST_IMAGE_PATTERN = "%s/{}/images/{}.png" % TEST_PATH


def read_img(img_id, flag_train=True):
    if flag_train:
        img_path = TRAIN_IMAGE_PATTERN.format(img_id, img_id)
    else:
        img_path = TEST_IMAGE_PATTERN.format(img_id, img_id)
    img = imread(img_path)
    img = img[:, :, :3]
    #r = scharr(img[:,:,0])
    #g = scharr(img[:,:,1])
    #b = scharr(img[:,:,2])
    #del img
    #return np.dstack([r,g,b])
    return img


def read_mask(mask_id, shape):
    mask_path = TRAIN_MASK_PATTERN.format(mask_id, mask_id)
    masks = imread_collection(mask_path).concatenate()
    height, width, _ = shape
    num_masks = masks.shape[0]
    mask = np.zeros((height, width), np.uint32)
    dmask = np.zeros((height, width), np.uint32)
    for index in range(0, num_masks):
        dm = scharr(masks[index])
        mask[masks[index] > 0] = 1
        dmask[dm > 0] = 1
    return np.dstack([mask, dmask])

def read_train_data(ids, cluster):
    train_img = []
    train_mask = []
    for img_id in ids:
        ti = read_img(img_id=img_id)
        tm = read_mask(mask_id=img_id, shape=ti.shape)
        train_img.append(ti)
        train_mask.append(tm)
    return train_img, train_mask

def read_test_data(ids):
    test_img = []
    for img_id in ids:
        ti = read_img(img_id=img_id, flag_train=False)
        test_img.append(ti)
    return test_img

def hor_flip(img):
    return img[::-1,:,:]

def vert_flip(img):
    return img[:,::-1,:]

def pader1(img, shape):
    el1 = img[:IMG_WIDTH//2, :IMG_HEIGHT//2, :]
    el2 = img[:IMG_WIDTH//2, :shape[1], :]
    el1 = np.hstack((el1, el2, el1))

    el2 = img[:shape[0], :IMG_HEIGHT//2, :]

    print(np.shape(img))
    img = np.hstack((el2, img, el2))
    print(np.shape(img))
    img = np.vstack((el1, img, el1))
    return img

def pader2(img, shape):
    x,y,c = shape
    tx, ty = x//IMG_WIDTH, y//IMG_HEIGHT

    img1 = np.hstack((img, img))
    img1 = np.vstack((img1, img1))

    img = img1[:IMG_WIDTH*(tx+1), :IMG_HEIGHT*(ty+1), :]
    return img

def slicer(img):
    shape = np.shape(img)
    if len(np.shape(img))==2:
        img = np.reshape(img, (shape[0], shape[1], 1))

    shape = np.shape(img)

    img_dict = {
        'img': [],
        #'vimg': [],
        #'himg': [],
        #'timg': [],
        #'vtimg': [],
        #'htimg': [],
        'pos': [],
        'shape': shape
    }

    img = pader2(img, shape)

    x,y,c = np.shape(img)
    tx, ty = x//IMG_WIDTH, y//IMG_HEIGHT

    himg = hor_flip(img)
    vimg = vert_flip(img)
    xflag = False
    for i in range(0, MERG_RATION*(tx)):
        if i//MERG_RATION >= tx-1 :
            sx1 = int(IMG_WIDTH*(i//MERG_RATION))
            sx2 = int(IMG_WIDTH*(i//MERG_RATION+1))
            xflag = True
        else:
            sx1 = int(IMG_WIDTH*(i/MERG_RATION))
            sx2 = int(IMG_WIDTH*(i/MERG_RATION+1))
        yflag = False
        for j in range(0, MERG_RATION*(ty)):
            if j//MERG_RATION >=ty-1 :
                sy1 = int(IMG_HEIGHT*(j//MERG_RATION))
                sy2 = int(IMG_HEIGHT*(j//MERG_RATION+1))
                yflag = True
            else:
                sy1 = int(IMG_HEIGHT*(j/MERG_RATION))
                sy2 = int(IMG_HEIGHT*(j/MERG_RATION+1))
            img_dict['pos'].append([i,j])
            #img_dict['vimg'].append(vimg[sx1:sx2, sy1:sy2, :])
            #img_dict['himg'].append(himg[sx1:sx2, sy1:sy2, :])
            img_dict['img'].append(img[sx1:sx2, sy1:sy2, :])
            #sx2, sx1 = x - sx1, x - sx2
            #sy2, sy1 = y - sy1, y - sy2
            #img_dict['vtimg'].append(vimg[sx1:sx2, sy1:sy2, :])
            #img_dict['htimg'].append(himg[sx1:sx2, sy1:sy2, :])
            #img_dict['timg'].append(img[sx1:sx2, sy1:sy2, :])
            if yflag:
                break
        if xflag:
            break
    return img_dict

def img_transform(imgs):
    img_set = []
    for img in imgs:
        img_set.append(slicer(img))
    return img_set

def set2list(imgs_set):
    imgs = []
    for iset in imgs_set:
        for l in iset:
            if 'img' in l:
                imgs.extend(iset[l])
    return imgs

def get_train_data(img_ids, cluster):
    tr_imgs, masks = read_train_data(img_ids, cluster)

    tr_imgs_set = img_transform(tr_imgs)
    del tr_imgs
    masks_set = img_transform(masks)
    del masks

    tr_imgs_list = set2list(tr_imgs_set)
    del tr_imgs_set
    masks_list = set2list(masks_set)
    del masks_set
    return tr_imgs_list, masks_list

def get_test_data(img_ids):
    te_imgs = read_test_data(img_ids)

    te_imgs_set = img_transform(te_imgs)
    return te_imgs_set

def image_ids_in(root_dir, ignore=[]):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids

def main():
    print('Read Path')
    #train_data = pd.read_csv('train.csv')
    #test_data = pd.read_csv('test.csv')
    train_image_ids = image_ids_in('input/train')
    #tr_ids = train_data.loc[train_data.hsv_cluster==3, 'image_id'].values
    #tr_ids = tr_ids[200:400]
    print('Get Data')
    X, y = get_train_data(train_image_ids, cluster=3)
    print(np.shape(X[0]))
    print(np.shape(y[0]))
    train_model(X, y, name='3')
    del X
    del y
    del tr_ids

    #te_ids = test_data.loc[:, 'image_id'].values
    #te_cluster = test_data.loc[:, 'hsv_cluster'].values
    #print(len(te_ids))
    #test_set = get_test_data(te_ids)
    #print(len(test_set))
    #test_model(test_set, te_ids, cluster=te_cluster)
    #"""

if __name__=='__main__':
    main()
