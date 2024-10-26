
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "src"))
from Data import DataSet

import unittest
import pandas
import numpy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TestDataSetPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample DataSet class (Assume DataSet class is imported from the actual code)
        self.dataSet = DataSet()

        # Create a sample DataFrame with missing values
        data = {
            'Feature_1': [1.0, 2.0, numpy.nan, 4.0, 5.0],
            'Feature_2': [7.0, 8.0, 9.0, numpy.nan, 10.0],
            'Dependent': [1, -1, 1, -1, numpy.nan]  # DV has missing value in last row
        }

        self.dataSet.dataFrame = pandas.DataFrame(data)

    def test_preprocessDataSet(self):
        # Run the preprocessing function
        self.dataSet.preprocessDataSet()

        # Test: check that the row with a missing dependent value is removed
        self.assertEqual(self.dataSet.dataFrame.shape[0], 4)

        # Test: check that missing independent values are filled with the mean
        mean_feature_1:float = self.dataSet.dataFrame['Feature_1'].mean()
        testTarget_feature_1:float = pandas.to_numeric(self.dataSet.dataFrame.loc[2, 'Feature_1'])
        self.assertAlmostEqual(
            testTarget_feature_1, 
            mean_feature_1
        )
        mean_feature_2 = self.dataSet.dataFrame['Feature_2'].mean()
        testTarget_feature_2: float = pandas.to_numeric(self.dataSet.dataFrame.loc[3, 'Feature_2'])
        self.assertAlmostEqual(
            testTarget_feature_2, 
            mean_feature_2
        )

        # Test: check the split into training and test sets
        self.assertEqual(self.dataSet.featureMatrix_train.shape[0], 3)
        self.assertEqual(self.dataSet.featureMatrix_test.shape[0], 1)

        # Test: check that feature scaling was applied to feature matrix training and test subsets
        featureMatrix_train_meanVals: numpy.ndarray = self.dataSet.featureMatrix_train.mean(axis=0)
        self.assertTrue(
            (numpy.abs(featureMatrix_train_meanVals) < 1e-5).all()
        )
        featureMatrix_train_stdDevVals: pandas.Series = self.dataSet.featureMatrix_train.std()
        self.assertTrue(
            (numpy.abs(featureMatrix_train_stdDevVals)-1 < 1e-5).all()
        )
        featureMatrix_test_meanVals: pandas.Series = self.dataSet.featureMatrix_test.mean(axis=0)
        self.assertTrue(
            (numpy.abs(featureMatrix_test_meanVals) < 1e-5).all()
        )
        featureMatrix_test_stdDevVals: pandas.Series = self.dataSet.featureMatrix_test.std()
        self.assertTrue(
            (numpy.abs(featureMatrix_test_stdDevVals)-1 < 1e-5).all()
        )


if __name__ == '__main__':
    unittest.main()
