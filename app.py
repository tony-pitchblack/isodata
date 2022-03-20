import isodata
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from help_funcs import img_to_ndarray

if __name__ == '__main__':
    # file = 'data/iris.data'
    # iris_data = np.loadtxt(file, delimiter=',')
    #
    # for line in iris_data:
    #     print(line)

    houses_img = Image.open('data/img/houses_grayscale.jpg')
    # houses_img = Image.open('data/img/houses.jpg')
    houses_data, N, M, depth = img_to_ndarray(houses_img)

    K = 2
    img_classes = isodata.isodata_classification(houses_data, parameters={'K': K})

    #show classes
    fig, ax = plt.subplots()
    im = ax.imshow(img_classes)

    ax.set_title(f"Satellite house image pixel classes ({K})")
    fig.tight_layout()
    plt.show()
    fig.savefig(f'data/img/houses_out_{K}_classes.jpg', bbox_inches='tight', dpi=500)

    houses_img.close()