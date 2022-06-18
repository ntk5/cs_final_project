from tensorflow import keras

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

DROPOUT_RATE = 0.5
DENSE_UNITS = 1
EPOCH_COUNT = 30
ACTIVATION_FUNCTION = 'sigmoid'

SHOULD_VERIFY_DEVICES = False

IS_LOCAL = False

if IS_LOCAL:
    DATA_DIR = r'C:\Users\nadav\Documents\unevrsity\thesis\datasets\skin\ISIC_2019_Training_Input'
    LABELS_PATH = r'C:\Users\nadav\Documents\unevrsity\thesis\datasets\skin\ISIC_2019_Training_GroundTruth.csv'
    print("Running in local env")
else:
    DATA_DIR = r'/home/nonadav4/datasets/ISIC_2019_Training_Input'
    LABELS_PATH = r'/home/nonadav4/datasets/ISIC_2019_Training_GroundTruth.csv'
    print("Running in server env")

# create metrices
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.BinaryAccuracy(name='val_accuracy'),
    keras.metrics.Precision(name='val_precision'),
    keras.metrics.Recall(name='val_recall'),
]
