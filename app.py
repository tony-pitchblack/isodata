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
    img_flat, N, M, depth = hf.img_to_ndarray(img)
    params = {
        'K': 3,
        # 'THETA_C': 100,
        # 'THETA_N': 200000
    }
    img_classes_flat, class_count = isodata_classification(img_flat, params)
    img_classes = img_classes_flat.reshape(N, M)

    # show clusterized image
    fig, ax = plt.subplots()
    im = ax.imshow(img_classes)

    # save clusterized image
    ax.set_title(f"Satellite house image pixel classes ({class_count})")
    fig.tight_layout()
    plt.show(block=True)
    fig.savefig(f"data/output/img/houses_out_{class_count}_classes.jpg", bbox_inches='tight', dpi=500)

    img.close()
