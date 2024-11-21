from Data import DataSet

import pandas
import numpy
from mpl_toolkits.mplot3d import Axes3D 
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as mat_plt


class Perceptron:
    def __init__(self, dataSet: DataSet = DataSet()) -> None:
        self.dataSet: DataSet = dataSet
        self.hyperPlane: numpy.ndarray = numpy.ndarray([])

    def executeLearningAlgorithm(self) -> None:
        # randomly initialize hyperplane equation
        w = numpy.random.rand(
            self.dataSet.featureMatrix.shape[1]
        )

        # train model
        misclassifiedDataPoints = self.makePredictions(
            self.dataSet.featureMatrix_train,
            self.dataSet.DependentVector_train,
            w
        )
        while misclassifiedDataPoints.any():
            x, expected_y = self.selectRandomMisclassifiedDataPoint(misclassifiedDataPoints)
            w = w + x * expected_y
            misclassifiedDataPoints = self.makePredictions(
                self.dataSet.featureMatrix_train,
                self.dataSet.DependentVector_train,
                w
            )

        # save output hyperplane
        self.hyperPlane = w

    def loadDataSet(self, dataSet: DataSet = DataSet()):
        self.dataSet = dataSet

    def getHyperPlane(self) -> numpy.ndarray:
        return self.hyperPlane

    def hypothesisFunction(self, x:numpy.ndarray, w:numpy.ndarray) -> numpy.float64:
        return numpy.sign(numpy.dot(w,x))

    def makePredictions(self, X, y, w) -> numpy.ndarray:
        predictions = numpy.apply_along_axis(
            self.hypothesisFunction,1,X,w
        )
        misclassifiedDataPoints = X[y != predictions]
        return misclassifiedDataPoints    
    
    # DEV NOTE: This is periodically causing a bug involving a missing index I think
    def selectRandomMisclassifiedDataPoint(self, misclassifiedDataPoints:numpy.ndarray) -> tuple[numpy.ndarray, float]:
        # select random 
        numpy.random.shuffle(misclassifiedDataPoints)
        x = misclassifiedDataPoints[0]
        indices = numpy.where(
            numpy.all(
                self.dataSet.featureMatrix_train == x, axis=1
            )
        )[0]
        index = indices[0]

        # DEBUGGING
        #print(type(index))
        #print(type(self.dataSet.DependentVector_train))
        #for indx, value in enumerate(self.dataSet.DependentVector_train):
        #    print(f"Index: {indx}, Value: {value}")
        #print("\n ----- \n")

        # DEBUGGING NOTE:
        # run time error is occuring here and may be due to fact that self.dataSet.DependentVector_train 
        # is a pandas Series instead of a numpy ndarray by the time execution reaches here.
        # pandas Series use label-based indexing, so if a series is replacing the expected numpy ndarray
        # this may explain why the code is breaking here

        y = float(self.dataSet.DependentVector_train[index])
        return x, y









"""# DEV NOTE: This is periodically causing a bug involving a missing index I think
    def selectRandomMisclassifiedDataPoint(self, misclassifiedDataPoints:numpy.ndarray) -> tuple[numpy.ndarray, float]:
        # select random 
        numpy.random.shuffle(misclassifiedDataPoints)
        x = misclassifiedDataPoints[0]
        y = float(
            self.dataSet.DependentVector_train[
                numpy.where(numpy.all(self.dataSet.featureMatrix_train == x, axis=1))[0]
            ]
        )
        return x, y"""
        
    