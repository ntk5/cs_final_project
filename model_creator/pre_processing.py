import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.transform import rotate
from tqdm import tqdm

from config import LABELS_PATH, DATA_DIR, IMG_HEIGHT, IMG_WIDTH


def create_image_generator():
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)


def create_dataframe():
    # Create data frames
    df = pd.read_csv(LABELS_PATH)
    # Add file suffix to be able to read
    df.image = df.image.astype(str) + '.jpg'
    # # casting to types
    df.MEL = df.MEL.astype(int)
    df.MEL = df.MEL.astype(str)
    df.SCC = df.SCC.astype(int)
    df.SCC = df.SCC.astype(str)
    return df


def augment_images(df):
    files_added = []
    print("Starting image augmentation:")
    for item in tqdm(df.query('SCC == "1"').image):
        img = cv2.imread(os.path.join(DATA_DIR, item))
        hflipped_image = np.fliplr(img)
        vflipped_image = np.flipud(img)
        r_image = rotate(img, angle=45, preserve_range=True)
        flipped_name = os.path.join(DATA_DIR, os.path.splitext(item)[0] + '_aug_s_1.jpg')
        vflipped_name = os.path.join(DATA_DIR, os.path.splitext(item)[0] + '_aug_s_2.jpg')
        rotated_name = os.path.join(DATA_DIR, os.path.splitext(item)[0] + '_aug_s_3.jpg')
        cv2.imwrite(flipped_name, hflipped_image)
        cv2.imwrite(vflipped_name, vflipped_image)
        cv2.imwrite(rotated_name, r_image)
        files_added += [flipped_name, vflipped_name, rotated_name]
    return files_added


def split_data(df):
    # get MEL data without SCC
    target_df = pd.concat([df.query('SCC == "1"'), df.query('SCC != "1"').head(5500)])
    source_df = df[~df.image.isin(list(target_df.image))]
    return source_df, target_df


def create_train_test_datagen(df_to_train, image_generator):
    source_df, target_df = split_data(df_to_train)
    # create image generators for MEL training
    train_data_gen = image_generator.flow_from_dataframe(
        dataframe=source_df,
        directory=DATA_DIR,
        x_col='image',
        y_col='MEL',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary')

    # create data get for scc
    scc_train, scc_test = np.split(target_df.sample(frac=1), [int(.8 * len(target_df))])
    scc_train_data_gen = image_generator.flow_from_dataframe(
        dataframe=scc_train,
        directory=DATA_DIR,
        x_col='image',
        y_col='SCC',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary')
    scc_test_data_gen = image_generator.flow_from_dataframe(
        dataframe=scc_test,
        directory=DATA_DIR,
        x_col='image',
        y_col='SCC',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=1,
        class_mode='binary')
    return train_data_gen, scc_train_data_gen, scc_test_data_gen


def create_labels(df, files_added_names):
    copy_df = df.copy()
    for item in tqdm(files_added_names):
        copy_df = copy_df.append(
            {'image': os.path.basename(item), 'MEL': '0', 'NV': '0', 'BCC': '0', 'AK': '0', 'BKL': '0', 'DF': '0',
             'VASC': '0', 'SCC': '1', 'UNK': '0'}, ignore_index=True)
    return copy_df
