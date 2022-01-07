from matplotlib import pyplot as plt


def show_batch(data_gen):
    image_batch, label_batch = next(data_gen)
    plt.figure(figsize=(10, 10))
    for n in range(32):
        ax = plt.subplot(8, 4, n + 1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title('SCC')
        else:
            plt.title('NOT-SCC')
        plt.axis('off')


def plot_metrics(history):
    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])

        plt.legend()


def visualize_model_training(model_history):
    plt.plot(model_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
