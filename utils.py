import matplotlib.pyplot as plt
import ripser
from gtda.diagrams import BettiCurve
import numpy as np
import os
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_custom(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Compute precision, recall, f1-score, and support for each class
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Compute overall accuracy
    accuracy = np.trace(cm) / float(np.sum(cm))

    # Compute true negative rate (TNR) for each class
    tnr = []
    for i in range(cm.shape[0]):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        tnr.append(tn / (np.sum(cm) - np.sum(cm[i, :])))

    # Return results as a dictionary
    results = {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "accuracy": accuracy,
        "f1-score": report["macro avg"]["f1-score"],
        "TNR": tnr,
    }

    return results

def plot_model_custom(model, name):
    plot_model(model,
               to_file=f"{os.getcwd()}/model_visualizations/{name}.png",
               show_shapes=True,
               rankdir="LR",
               expand_nested=True,
               show_layer_names=True)


def visualize_history(hist, name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(hist.history['accuracy'])
    ax1.plot(hist.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(hist.history['loss'])
    ax2.plot(hist.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    plt.show()

    print(f'{name} model accuracy (val): {hist.history["val_accuracy"][-1]}')


def transform_images_to_betti_curves(images) -> np.array:
    """Gets images from Keras generator
    and return betti curve in R^100"""

    # The flow_from_directory method with color_mode='grayscale' returns a 3D array
    vectorized_images = np.array(
        images[:len(images)]
    ).reshape(-1, 128, 128) / 255

    dgms = [ripser.lower_star_img(-img) for img in vectorized_images]

    dgms_processed = [
        [[(pt[0], pt[1], 0) for pt in dgm[:-1]]]
        for dgm in dgms
    ]

    bc = BettiCurve()

    betti_curves = np.array([
        bc.fit_transform(X=dgm)[0][0]
        for dgm in dgms_processed
    ])

    return betti_curves
