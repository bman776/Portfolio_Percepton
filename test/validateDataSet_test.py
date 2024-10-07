import random
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "src"))
import GUI
import unittest
import matplotlib.pyplot
import numpy
import csv
import itertools

from typing import Iterator, Any, List
from pathlib import Path

dataDirectory = os.path.join(os.path.dirname(sys.path[0]), "res")

class Test_validateDataSet_Function(unittest.TestCase):

    csv_invalidityTypeDict = {
        "finalFeatureNotBinaryCategorical": 1,
        "nonFinalFeatureIsNonNumeric": 2,
        "nonFinalFeatureIsBinaryCategorical": 3
    }

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.testGUI: GUI.GUI = GUI.GUI()
        
    @classmethod
    def generateRandomValid_CSV(cls, fileName: str, num_rows: int, num_cols: int) -> None:
        with open(fileName, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # write header
            header = [f'Feature_{i+1}' for i in range(num_cols)]
            csv_writer.writerow(header)

            # write in random rows of data according to format 
            for _ in range(num_rows):

                # Generate num_cols-1 real number values
                realFeatureVals = [random.uniform(0,100) for _ in range(num_cols-1)]

                # Generate a random value from set [-1,1]
                binCatFeatureVal = random.choice([-1,1])

                csv_writer.writerow(realFeatureVals + [binCatFeatureVal])



    def test_validDataSetProvided(self):
        testDataSetPath = os.path.join(dataDirectory, "validDataSet1.csv")
        self.generateRandomValid_CSV(testDataSetPath, 5, 5)
        self.assertTrue(
            self.testGUI.validateDataSet(testDataSetPath)
        )

        testDataSetPath = os.path.join(dataDirectory, "validDataSet2.csv")
        self.generateRandomValid_CSV(testDataSetPath, 20, 5)
        self.assertTrue(
            self.testGUI.validateDataSet(testDataSetPath)
        )

        testDataSetPath = os.path.join(dataDirectory, "validDataSet3.csv")
        self.generateRandomValid_CSV(testDataSetPath, 100, 6)
        self.assertTrue(
            self.testGUI.validateDataSet(testDataSetPath)
        )


    def test_invalidDataSetProvided_finalFeatureNotBinaryCategorical(self):
        testDataSetPath = os.path.join(dataDirectory, "invalidDataSet1_finalFeatureNotBinaryCategorical.csv")
        self.assertFalse(
            self.testGUI.validateDataSet(testDataSetPath)
        )

    def test_invalidDataSetProvided_nonFinalFeatureIsNonNumeric(self):
        testDataSetPath = os.path.join(dataDirectory, "invalidDataSet2_nonFinalFeatureIsNonNumeric.csv")
        self.assertFalse(
            self.testGUI.validateDataSet(testDataSetPath)
        )

if __name__ == "__main__":
    unittest.main()


"""
JUNK CODE:

@classmethod
    def generateRandomInvalid_CSV(cls, fileName: str, num_rows: int, num_cols: int) -> None:
        with open(fileName, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # write header
            header = [f'Feature_{i+1}' for i in range(num_cols)]
            csv_writer.writerow(header)

            # randomly decide 1 of 3 ways the csv file will be invalid
            csv_invalidityChoice = random.choice([0,1,2])
            if csv_invalidityChoice == cls.csv_invalidityTypeDict["finalFeatureNotBinaryCategorical"]:

                # randomly decide one of three ways the final feature will not be a binary categorical in [-1,1]
                if random.choice([0,1]) == 0:
                    # final feature will be populated with real number values
                    for _ in range(num_rows):
                        csv_writer.writerow([random.uniform(0,100) for _ in range(num_cols)])
                else:
                    pass
                    

                # write in random rows of data according to format 
                for _ in range(num_rows):

                    # Generate num_cols # of real number values
                    realFeatureVals = [random.uniform(0,100) for _ in range(num_cols)]

                    csv_writer.writerow(realFeatureVals + [binCatFeatureVal])

            elif csv_invalidityChoice == cls.csv_invalidityTypeDict["nonFinalFeatureIsNonNumeric"]:
                pass
            elif csv_invalidityChoice == cls.csv_invalidityTypeDict["nonFinalFeatureIsBinaryCategorical"]:
                pass
            

            # write in random rows of data according to format 
            for _ in range(num_rows):

                # Generate num_cols-1 real number values
                realFeatureVals = [random.uniform(0,100) for _ in range(num_cols-1)]

                # Generate a random value from set [-1,1]
                binCatFeatureVal = random.choice([-1,1])

                csv_writer.writerow(realFeatureVals + [binCatFeatureVal])
"""