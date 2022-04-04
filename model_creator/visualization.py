import matplotlib
from matplotlib import pyplot as plt

from pre_processing import create_dataframe


def show_classes_histogram():
    df = create_dataframe()
    for i in df.columns:
        if i != 'image':
            df[i] = df[i].astype(int)
    del df['image']
    del df['UNK']

    s = df.sum()

    s.plot.pie(y=df.index,
               explode=(0, 0, 0, 0, 0, 0, 0, 0.15),  # exploding 'SCC'
               autopct='%1.1f%%')

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def show_batch(data_gen):
    image_batch, label_batch = next(data_gen)
    plt.figure(figsize=(10, 10))
    for n in range(32):
        plt.subplot(8, 4, n + 1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title('SCC')
        else:
            plt.title('NOT-SCC')
        plt.axis('off')


def plot_metrics(history, is_base=False):
    plt.clf()
    matplotlib.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{"base_" if is_base else ""}accuracy.png')
    plt.show()
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{"base_" if is_base else ""}loss.png')
    plt.show()
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('model recall')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{"base_" if is_base else ""}recall.png')
    plt.show()
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{"base_" if is_base else ""}precision.png')
    plt.show()
    plt.clf()


def visualize_model_training(model_history, is_base=False):
    plt.plot(model_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{"base_" if is_base else ""}accuracy_training.png')
    plt.show()
    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{"base_" if is_base else ""}loss_training.png')
    plt.show()
