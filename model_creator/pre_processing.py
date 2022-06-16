import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
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
        with Image.open(os.path.join(DATA_DIR, item)) as curr_img:
            img = np.array(curr_img)
            # Apply Image Flipping horizontally and vertically
            horizontally_flipped_image = Image.fromarray(np.fliplr(img))
            veritcally_flipped_image = Image.fromarray(np.flipud(img))
            # Apply Image Rotation by 45
            rotated_image = Image.fromarray(np.flipud(img))
            rotated_image = rotated_image.rotate(45)

            # Rename and save the new images
            flipped_name = os.path.join(DATA_DIR, os.path.splitext(item)[0] + '_aug_s_1.jpg')
            vflipped_name = os.path.join(DATA_DIR, os.path.splitext(item)[0] + '_aug_s_2.jpg')
            rotated_name = os.path.join(DATA_DIR, os.path.splitext(item)[0] + '_aug_s_3.jpg')
            horizontally_flipped_image.save(flipped_name)
            veritcally_flipped_image.save(vflipped_name)
            rotated_image.save(rotated_name)

            files_added += [flipped_name, vflipped_name, rotated_name]
    return files_added


def split_data(df):
    # get MEL data without SCC
    target_df = pd.concat([df.query('SCC == "1"'), df.query('SCC != "1"').head(5500)])
    source_df = df[~df.image.isin(list(target_df.image))]
    return source_df, target_df


def create_train_test_datagen(image_generator, source_df, scc_train, scc_validation, scc_test):
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
    scc_train_data_gen = image_generator.flow_from_dataframe(
        dataframe=scc_train,
        directory=DATA_DIR,
        x_col='image',
        y_col='SCC',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary')

    scc_validation_data_gen = image_generator.flow_from_dataframe(
        dataframe=scc_validation,
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
    return train_data_gen, scc_train_data_gen, scc_test_data_gen, scc_validation_data_gen


def create_labels(df, files_added_names):
    copy_df = df.copy()
    for item in tqdm(files_added_names):
        copy_df = copy_df.append(
            {'image': os.path.basename(item), 'MEL': '0', 'NV': '0', 'BCC': '0', 'AK': '0', 'BKL': '0', 'DF': '0',
             'VASC': '0', 'SCC': '1', 'UNK': '0'}, ignore_index=True)
    return copy_df
