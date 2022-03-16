import isodata
import numpy as np
from PIL import Image

if __name__ == '__main__':
    #import data
    #file = 'data/iris.data'
    #iris_data = np.loadtxt(file, delimiter=',')

    #for line in iris_data:
    #    print(line)

    houses_img = Image.open('data/img/houses.jpg')
    houses_data = np.asarray(houses_img)
    print(houses_data.shape)
    isodata.isodata_classification(houses_data)