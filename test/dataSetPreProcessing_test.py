
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "src"))
from Data import DataSet

import unittest
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TestDataSetPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample DataSet class (Assume DataSet class is imported from the actual code)
        self.dataSet = DataSet()

        # Create a sample DataFrame with missing values
        data = {
            'Feature_1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'Feature_2': [7.0, 8.0, 9.0, 10.0, np.nan],
            'Dependent': [1, -1, 1, -1, np.nan]  # DV has missing value in last row
        }

        self.dataSet.dataFrame = pd.DataFrame(data)

    def test_preprocessDataSet(self):
        # Run the preprocessing function
        self.dataSet.preprocessDataSet()

        # Test: check that the row with a missing dependent value is removed
        self.assertEqual(self.dataSet.dataFrame.shape[0], 4)

        # Test: check that missing independent values are filled with the mean
        mean_feature_1 = self.dataSet.dataFrame['Feature_1'].mean()
        testTarget_feature_1 = float(self.dataSet.dataFrame.loc[2, 'Feature_1'])
        self.assertAlmostEqual(
            testTarget_feature_1, 
            mean_feature_1
        )

        mean_feature_2 = self.dataSet.dataFrame['Feature_2'].mean()
        self.assertAlmostEqual(self.dataSet.dataFrame.loc[4, 'Feature_2'], mean_feature_2)

        # Test: check the split into training and test sets
        self.assertEqual(self.dataSet.featureMatrix_train.shape[0], 3)
        self.assertEqual(self.dataSet.featureMatrix_test.shape[0], 1)

        # Test: check that feature scaling was applied
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.dataSet.dataFrame[['Feature_1', 'Feature_2']])
        np.testing.assert_array_almost_equal(self.dataSet.featureMatrix_train, X_train_scaled[:3])

if __name__ == '__main__':
    unittest.main()
