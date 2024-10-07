import pandas
import numpy
from mpl_toolkits.mplot3d import Axes3D 
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as mat_plt


class Perceptron:
    def __init__(self, ds: pandas.DataFrame) -> None:
        self.dataSet: pandas.DataFrame = ds
        self.hyperPlane: numpy.ndarray = numpy.ndarray([])
        
        
        pass