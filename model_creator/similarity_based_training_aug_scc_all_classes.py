# for paths

# for plots
# for getting images from path
import tensorflow as tf

from config import SHOULD_VERIFY_DEVICES
from model_creating import train_base_model, evaluate_model, train_clean_model
from pre_processing import create_image_generator, create_dataframe, augment_images, \
    create_train_test_datagen, create_labels
from visualization import show_batch, plot_metrics


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
    # rescale images
    image_generator = create_image_generator()
    df = create_dataframe()
    files_added_names = augment_images(df)
    extended_df = create_labels(df, files_added_names)
    train_data_gen, scc_train_data_gen, scc_test_data_gen = create_train_test_datagen(extended_df, image_generator)
    show_batch(scc_train_data_gen)
    # resnet50 without top with weights pre-trained on imagenet
    pretrained_base_model, base_model_history = train_base_model(scc_train_data_gen, scc_test_data_gen)
    evaluate_model(pretrained_base_model, scc_test_data_gen)
    plot_metrics(base_model_history, is_base=True)
    # summarize history for accuracy
    new_model, history = train_clean_model(train_data_gen, scc_train_data_gen, scc_test_data_gen)
    evaluate_model(new_model, scc_test_data_gen)
    plot_metrics(history)


if __name__ == '__main__':
    main()
