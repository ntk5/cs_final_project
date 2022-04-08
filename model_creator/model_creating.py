import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.applications.resnet import ResNet50

from config import IMG_WIDTH, IMG_HEIGHT, METRICS, DROPOUT_RATE, DENSE_UNITS, ACTIVATION_FUNCTION, EPOCH_COUNT


def train_clean_model(base_data_gen, target_data_gen, validation_data):
    resnet50 = ResNet50(weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    resnet50.trainable = True

    model = keras.Sequential()
    model.add(resnet50)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(DROPOUT_RATE))
    model.add(keras.layers.Dense(units=DENSE_UNITS, activation=ACTIVATION_FUNCTION))
    model.summary()

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)

    steps_per_epoch = base_data_gen.n // base_data_gen.batch_size
    model.fit(x=base_data_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCH_COUNT)

    model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = -3

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    new_model = keras.Sequential()
    new_model.add(model)
    new_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)
    new_model.summary()
    steps_per_epoch = target_data_gen.n // target_data_gen.batch_size
    history = new_model.fit(x=target_data_gen, validation_data=validation_data, steps_per_epoch=steps_per_epoch,
                            epochs=EPOCH_COUNT)

    save_model(new_model, 'new_model')

    return new_model, history


def train_base_model(train_data_gen, validation_data):
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    # freeze all current layers
    resnet50.trainable = False
    # add fine-tuning layers
    model = keras.Sequential()
    model.add(resnet50)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(DROPOUT_RATE))
    model.add(keras.layers.Dense(units=DENSE_UNITS, activation=ACTIVATION_FUNCTION))
    model.summary()

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)

    steps_per_epoch = train_data_gen.n // train_data_gen.batch_size
    model_history = model.fit(x=train_data_gen, validation_data=validation_data, steps_per_epoch=steps_per_epoch,
                              epochs=EPOCH_COUNT)

    save_model(model, 'base_model')

    return model, model_history


def save_model(model, description):
    ts = int(time.time())
    file_path = f"./scc_classifier/{description}_{ts}/"
    model.save(filepath=file_path, save_format='tf')


def evaluate_model(model_to_evaluate, data_gen):
    benchmark_model_start_time = time.time()

    res = model_to_evaluate.evaluate(x=data_gen, verbose=2)

    benchmark_model_end_time = time.time()

    total_benchmark_model_time = benchmark_model_end_time - benchmark_model_start_time
    print("Total inference time: {total}".format(total=total_benchmark_model_time))
    print("Average inference time: {total}".format(total=total_benchmark_model_time / data_gen.n))
    print("Benchmark Model evaluating: {res}".format(res=res))
    return res
