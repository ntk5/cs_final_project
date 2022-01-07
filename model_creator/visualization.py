import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


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


def plot_metrics(history, ground_truth, predicted_res, is_base=False):
    matplotlib.rcParams['figure.figsize'] = (12, 10)
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

        if is_base:
            plt.savefig(f'{metric}_base.png')
        else:
            plt.savefig(f'{metric}.png')

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(ground_truth.classes[:9], predicted_res)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    if is_base:
        plt.savefig(f'roc_curve_base.png')
    else:
        plt.savefig(f'roc_curve.png')


def visualize_model_training(model_history, is_base=False):
    plt.plot(model_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if is_base:
        plt.savefig('accuracy_training_base.png')
    else:
        plt.savefig('accuracy_training.png')
    plt.show()
    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if is_base:
        plt.savefig('loss_training_base.png')
    else:
        plt.savefig('loss_training.png')
    plt.show()
