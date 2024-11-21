import pandas
import numpy

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class DataSet:
    def __init__(self, data: pandas.DataFrame = pandas.DataFrame()) -> None:
        self.dataFrame: pandas.DataFrame = data

        self.featureMatrix: pandas.DataFrame = pandas.DataFrame()
        self.DependentVector: numpy.ndarray = numpy.ndarray([]) 
        self.featureMatrix_train: numpy.ndarray = numpy.ndarray([])
        self.featureMatrix_test: numpy.ndarray = numpy.ndarray([])
        self.DependentVector_train: numpy.ndarray = numpy.ndarray([])
        self.DependentVector_test: numpy.ndarray = numpy.ndarray([])

        self.numericImputer: SimpleImputer = SimpleImputer(missing_values=numpy.nan, strategy="mean")
        self.stdScaler_train: StandardScaler = StandardScaler()
        self.stdScaler_test: StandardScaler = StandardScaler()

    def loadData(self, data: pandas.DataFrame = pandas.DataFrame()) -> None:
        self.dataFrame = data

    def getFeatureDimensions(self) -> int:
        if not(self.featureMatrix.empty):
            return self.featureMatrix.shape[1]
        else:
            return -1

    # DEV NOTE:
    # This function both validates the data set and ensures its valid values are converted to numeric data type
    # EDIT: Not sure if the dataframe contents are in fact being converted to numeric here
    def validateDataSet(self) -> bool:
        # Check if last column is binary categorical feature in values [-1, 1]
        lastFeatureValid: bool = self.dataFrame.iloc[:,-1].isin([-1,1]).all()

        # Check if all columns except the last contain only numeric value
        remFeaturesValid: bool = True
        for col in self.dataFrame.columns[:-1]:
            # check if current column is numeric by attempting to convert its string values to numeric values 
            try:
                self.dataFrame[col] = pandas.to_numeric(self.dataFrame[col], errors='raise')
            except ValueError as e:
                remFeaturesValid = False
                break

        if not (lastFeatureValid and remFeaturesValid):
            return False
        
        return True
    
    # DEV NOTE:
    # The data set could be quite large, should have data preprocessing functions modify the data set in place
    # instead of modifying and returning copies 

    def preprocessDataSet(self) -> None:
        self.addressMissingDV_vals_rowRemoval()
        self.addressMissingIV_vals_meanInsert()
        self.test_train_splitDataSet()
        self.featureScale_trainingSet()

    def addressMissingDV_vals_rowRemoval(self) -> None:
        self.dataFrame.dropna(
            subset=[self.dataFrame.columns[-1]], inplace=True
        )

    def addressMissingIV_vals_meanInsert(self) -> None:
        numericImputer = SimpleImputer(missing_values=numpy.nan, strategy="mean")
        numericImputer.fit(self.dataFrame[self.dataFrame.columns[:-1]])
        self.dataFrame[self.dataFrame.columns[:-1]] = numericImputer.transform(
            self.dataFrame[self.dataFrame.columns[:-1]]
        )

    def test_train_splitDataSet(self) -> None:
        self.featureMatrix = self.dataFrame.iloc[:,:-1]
        # DEV NOTE:
        # .iloc() function returns a series when a single row or col is being extracted from a dataframe
        self.dependentVector = self.dataFrame.iloc[:,-1].to_numpy()
        self.featureMatrix_train, self.featureMatrix_test, self.DependentVector_train, self.DependentVector_test = train_test_split(
            self.featureMatrix,
            self.dependentVector,
            test_size=0.2,
            random_state=0
        )
       
    
    def featureScale_trainingSet(self) -> None:
        self.stdScaler_train.fit(self.featureMatrix_train)
        self.stdScaler_test.fit(self.featureMatrix_test)
        self.featureMatrix_train = self.stdScaler_train.transform(self.featureMatrix_train)
        self.featureMatrix_test = self.stdScaler_test.transform(self.featureMatrix_test)