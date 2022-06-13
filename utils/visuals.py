import numpy as np
import matplotlib.pyplot as plt


def show_batch_images(images):
    batch_size = images.shape[0]
    fig, axes = plt.subplots(int(batch_size/4), 4, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(np.transpose(images[i], (1, 2, 0)), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()