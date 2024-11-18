import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "src"))
from Data import DataSet
from Model import Perceptron

import unittest
import pandas
import numpy

class TestModelBuilding(unittest.TestCase):

    def setUp(self):
        data = {
            'Feature_1': [8,4,9,7,9,4,10,2,8,7,4,4,1,2],
            'Feature_2': [7,10,7,10,6,8,10,7,3,5,4,6,3,5],
            'Dependent': [1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1]
        }

        # DEV NOTE:
        # Need to have inner Data set object call preprocessDataSet() to 
        # initialize feature matrix and dependent vector attributes else 
        # these attributes will be empty when perceptron object calls
        # executeLearningAlgorithm() which operates with them
        dataset = DataSet(pandas.DataFrame(data))
        dataset.preprocessDataSet()
        self.perceptronModel = Perceptron(dataset)


    def test_modelOnTrainingData(self):
        # Build Model
        self.perceptronModel.executeLearningAlgorithm()

        # get Model
        modelHyperplane = self.perceptronModel.getHyperPlane()

        # iterate through rows of data set and input each into Model
        # compare actual output to expected output
        # any misclassifications indicate error in model building logic since model
        # is trained on this data and therefore if built correctly cannot
        # miscalssify the same data it was trained on
        misclassifiedData = self.perceptronModel.makePredictions(
                self.perceptronModel.dataSet.featureMatrix_train,
                self.perceptronModel.dataSet.DependentVector_train,
                modelHyperplane
        )
        self.assertEqual(misclassifiedData.size,0)


if __name__ == '__main__':
    unittest.main()


