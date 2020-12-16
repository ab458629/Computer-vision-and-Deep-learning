from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import cv2


def createLabels(data):
    for item in data:
        height = item.get_height()
        plt.text(
            item.get_x()+item.get_width()/2.,
            height,
            height,
            ha="center",
            va="bottom",
        )


if __name__ == '__main__':
    cv2.namedWindow("TensorBoard", 0)
    cv2.resizeWindow("TensorBoard", 1280, 720)
    img = cv2.imread("Resnet50_tensorboard.PNG")
    cv2.imshow("TensorBoard", img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    # C:\>tensorboard --logdir="C:\Computer vision and Deep learning\HW2\logs" --host=127.0.0.1

    cat_or_dog = ['Cats', 'Dogs']

    net = tf.contrib.keras.models.load_model('HW02_05_ResNet50.h5')

    pathDir = os.listdir("data/validation/cat")
    filename = random.sample(pathDir, 1)

    img = image.load_img("data/validation/cat/" +
                         filename[0], target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # 記得擴充維度
    pred = net.predict(x)[0]
    pred_index = np.argmax(pred)

    plt.imshow(img)
    plt.title('Class : ' + cat_or_dog[pred_index])
    plt.show()

    col_count = 2
    bar_width = 0.2
    index = np.arange(col_count)

    # 這裡我不知道為甚麼使用random-erasing後的accuracy反而下降

    Before = [0.9431, 0.969]
    After = [0.8468, 0.872]

    A = plt.bar(index,
                Before,
                bar_width,
                alpha=.4,
                label="Before")
    B = plt.bar(index+0.2,
                After,
                bar_width,
                alpha=.4,
                label="After")

    createLabels(A)
    createLabels(B)

    plt.ylabel("Accuracy")
    plt.title("Before/After Random erasing")
    plt.xticks(index+.3 / 2, ("Traing", "Validation"))
    plt.legend()
    plt.grid(True)
    plt.show()
