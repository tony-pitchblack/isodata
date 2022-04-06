from isodata import isodata_classification
import numpy as np
import help_funcs as hf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':
    # load and show test image

    # img = Image.open('data/input/img/houses_grayscale.jpg')
    # imgplot = plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    img = Image.open('data/input/img/houses.jpg')
    imgplot = plt.imshow(img)

    # clusterize using ISODATA algorithm
    houses_data, N, M, depth = hf.img_to_ndarray(img)
    params = {
        'K': 3,
        'THETA_C': 100,
        'THETA_N': 200000
    }
    img_classes = isodata_classification(houses_data, params)

    # show clusterized image

    fig, ax = plt.subplots()
    im = ax.imshow(img_classes)

    # save clusterized image
    ax.set_title(f"Satellite house image pixel classes ({K})")
    fig.tight_layout()
    plt.show(block=True)
    # fig.savefig(f'data/output/img/houses_out_{K}_classes.jpg', bbox_inches='tight', dpi=500)

    img.close()
