from PIL import Image
from random import shuffle

import json
import os
import glob
import sys
import shutil
import math
import time

import numpy as np

dir_training_pool = '/home/alvinli/Desktop/EF/dataset/EF-training-Pool'
dir_batches = os.path.join(dir_training_pool, 'batches')
dir_train_batch = os.path.join(dir_batches, 'train_batch')
dir_test_batch = os.path.join(dir_batches, 'test_batch')
dir_datasheet = os.path.join(dir_training_pool, 'datasheet.json')

ratio_of_separate = 0.8
IMAGE_SIZE = (800, 600)
Img_width, Img_height = IMAGE_SIZE

global datasheet
global index_card

def main():
    
    global datasheet
    global index_card

    os.mkdir(dir_batches)
    
    try:
        with open(dir_datasheet) as json_file:
            datasheet = json.load(json_file)
    except IOError:
        print('An error occured trying to read {}.'.format(dir_datasheet))

    Box = list(datasheet.keys())
    Box_size = len(Box)
    for k in range(Box_size):
        N = len(datasheet[Box[k]])
        if N != 0:
            try:
                index_card = [i for i in range(N)]
                shuffle(index_card)
                get_batch_for_train(Box[k]) 
                get_batch_for_eval(Box[k])
            except:
                print("Unexpected error:", sys.exc_info()[0])

        else:
            print('{} has nothing stored in {}.'.format(Box[k], dir_datasheet))

def get_batch_for_train(K):
    "do something"
    global datasheet
    global index_card

    os.mkdir(dir_train_batch)

    num_example_per_epoch_for_train = math.ceil(len(datasheet[K]) * ratio_of_separate)
    InnerKeys = list(datasheet[K])
    for i in index_card[:num_example_per_epoch_for_train]:
        train_img = InnerKeys[i]
        train_msk = datasheet[K][InnerKeys[i]]
        try:
            img = Image.open(train_img)
            msk = Image.open(train_msk)
            if img.size != msk.size:
                print('The input image {} and its mask {} are unmatched.'.format(train_img, train_msk))
                break
            else:
                if img.size == (IMAGE_SIZE):
                    img_path, img_name = os.path.split(train_img)
                    msk_path, msk_name = os.path.split(train_msk)
                    shutil.copy(train_img, os.path.join(dir_train_batch, img_name))
                    shutil.copy(train_msk, os.path.join(dir_train_batch, msk_name))
                else:
                    print("Resize this image and its mask with the original apsect ratio preserved.")
                    width, height = img.size 
                    img.thumbnail(IMAGE_SIZE)
                    msk.thumbnail(IMAGE_SIZE)
                    arr_img = np.asarray(img)
                    arr_msk = np.asarray(msk)
                    new_img = np.zeros(IMAGE_SIZE)
                    new_msk = np.zeros(IMAGE_SIZE)

                    if width < Img_width:
                        shift_start = math.ceil( (Img_width - width) / 2 )
                        shift_end = shift_start + width
                        new_img[shift_start:shift_end, :] = arr_img
                        new_msk[shift_start:shift_end, :] = arr_msk
                    
                    if height < Img_height:
                        shift_start = math.ceil( (Img_height - height) / 2 )
                        shift_end = shift_start + height
                        new_img[:, shift_start:shift_end] = arr_img
                        new_msk[:, shift_start:shift_end] = arr_msk

                    new_img = Image.fromarray(np.uint8(new_img))
                    new_msk = Image.fromarray(np.uint8(new_msk))
                    
                    img_path, img_name = os.path.split(train_img)
                    msk_path, msk_name = os.path.split(train_msk)
                    new_img.save(os.path.join(dir_train_batch, img_name))
                    new_msk.save(os.path.join(dir_train_batch, msk_name))

        except:
            print("Unexpected error:", sys.exc_info()[0])


def get_batch_for_eval(K):
    "do something"
    global datasheet
    global index_card

    os.mkdir(dir_test_batch)

    num_example_per_epoch_for_eval = len(datasheet[K]) -  math.ceil(len(datasheet[K]) * ratio_of_separate)
    InnerKeys = list(datasheet[K])
    for i in index_card[-num_example_per_epoch_for_eval:]:
        eval_img = InnerKeys[i]
        eval_msk = datasheet[K][InnerKeys[i]]
        try:
            img = Image.open(eval_img)
            msk = Image.open(eval_msk)
            if img.size != msk.size:
                print('The input image {} and its mask {} are unmatched.'.format(eval_img, eval_msk))
                break
            else:
                if img.size == (IMAGE_SIZE):
                    img_path, img_name = os.path.split(eval_img)
                    msk_path, msk_name = os.path.split(eval_msk)
                    shutil.copy(eval_img, os.path.join(dir_test_batch, img_name))
                    shutil.copy(eval_msk, os.path.join(dir_test_batch, msk_name))
                else:
                    print("Resize this image and its mask with the original apsect ratio preserved.")
                    width, height = img.size 
                    img.thumbnail(IMAGE_SIZE)
                    msk.thumbnail(IMAGE_SIZE)
                    arr_img = np.asarray(img)
                    arr_msk = np.asarray(msk)
                    new_img = np.zeros(IMAGE_SIZE)
                    new_msk = np.zeros(IMAGE_SIZE)

                    if width < Img_width:
                        shift_start = math.ceil( (Img_width - width) / 2 )
                        shift_end = shift_start + width
                        new_img[shift_start:shift_end, :] = arr_img
                        new_msk[shift_start:shift_end, :] = arr_msk
                    
                    if height < Img_height:
                        shift_start = math.ceil( (Img_height - height) / 2 )
                        shift_end = shift_start + height
                        new_img[:, shift_start:shift_end] = arr_img
                        new_msk[:, shift_start:shift_end] = arr_msk

                    new_img = Image.fromarray(np.uint8(new_img))
                    new_msk = Image.fromarray(np.uint8(new_msk))
                    
                    img_path, img_name = os.path.split(eval_img)
                    msk_path, msk_name = os.path.split(eval_msk)
                    new_img.save(os.path.join(dir_test_batch, img_name))
                    new_msk.save(os.path.join(dir_test_batch, msk_name))

        except:
            print("Unexpected error:", sys.exc_info()[0])       


if __name__ == '__main__':
#    start = time. time()
    main()
#    end = time. time()

#    duration = end - start
#    print('\nThis code runs so fast that only spends {} in second.'.format(duration))

