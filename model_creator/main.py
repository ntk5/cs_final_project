# for paths

# for plots
# for getting images from path
import numpy as np
import tensorflow as tf

from config import SHOULD_VERIFY_DEVICES
from model_creating import train_pre_trained_model, evaluate_model, train_clean_model
from pre_processing import create_image_generator, create_dataframe, augment_images, \
    create_train_test_datagen, create_labels, split_data
from visualization import plot_metrics


# working with keras
# We are testing ResNet50 Model


def verify_tf_env():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    # verify all devices are recognized
    print(tf.config.list_physical_devices())
    # verify tf version
    print(tf.__version__)


def main():
    if SHOULD_VERIFY_DEVICES:
        verify_tf_env()

    image_generator = create_image_generator()
    df = create_dataframe()
    source_df, target_df = split_data(df)
    scc_train, scc_validation, scc_test = np.split(target_df.sample(frac=1, random_state=42),
                                                   [int(.8 * len(target_df)), int(.9 * len(target_df))])

    # ====================================== stuff without aug ===========================

    print("Creating non augmentation datagen")
    train_data_gen_no_aug, scc_train_data_gen_no_aug, scc_test_data_gen_no_aug, scc_validation_data_gen_no_aug = create_train_test_datagen(
        image_generator, source_df, scc_train, scc_validation, scc_test)

    # train ImageNet
    print("Training imagenet model with no scc aug")
    ImageNet_model, ImageNet_model_history = train_pre_trained_model(train_data_gen_no_aug,
                                                                     scc_validation_data_gen_no_aug)
    print("evaluating imageNet no Scc aug")
    evaluate_model(ImageNet_model, scc_test_data_gen_no_aug)
    plot_metrics(ImageNet_model_history, model_name="ImageNet")

    # train ISIC-SCC
    print("Training ISIC_MEL no SCC aug")
    ISIC_MEL_model, ISIC_MEL_model_history = train_clean_model(train_data_gen_no_aug, scc_train_data_gen_no_aug,
                                                               scc_validation_data_gen_no_aug)
    print("evaluating ISIC_MEL no SCC aug")
    evaluate_model(ISIC_MEL_model, scc_test_data_gen_no_aug)
    plot_metrics(ISIC_MEL_model_history, model_name="ISIC_MEL")

    # ===================================== aug stuff ==================================

    print("perform augmentation")
    files_added_names = augment_images(scc_train)
    extended_scc_train_df = create_labels(scc_train, files_added_names)

    print("Creating augmentation datagen")
    train_data_gen, scc_train_data_gen, scc_test_data_gen, scc_validation_data_gen = create_train_test_datagen(
        image_generator, source_df, extended_scc_train_df, scc_validation, scc_test)

    # train ImageNet augmented SCC
    print("train ImageNet augmented SCC")
    ImageNet_augmented_SCC_model, ImageNet_augmented_SCC_model_history = train_pre_trained_model(scc_train_data_gen,
                                                                                                 scc_validation_data_gen)
    print("Evaluate ImageNet augmented SCC")
    evaluate_model(ImageNet_augmented_SCC_model, scc_test_data_gen)
    plot_metrics(ImageNet_augmented_SCC_model_history, model_name="ImageNet_augmented_SCC")

    # train ISIC-MEL augmented SCC
    print("train ISIC-MEL augmented SCC")
    ISIC_MEL_augmented_SCC_model, ISIC_MEL_augmented_SCC_model_history = train_clean_model(train_data_gen,
                                                                                           scc_train_data_gen,
                                                                                           scc_validation_data_gen)
    print("evaluate ISIC-MEL augmented SCC")
    evaluate_model(ISIC_MEL_augmented_SCC_model, scc_test_data_gen)
    plot_metrics(ISIC_MEL_augmented_SCC_model_history, model_name="ISIC_MEL_augmented_SCC")


if __name__ == '__main__':
    main()
