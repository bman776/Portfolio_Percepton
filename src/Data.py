import pandas
import numpy

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class DataSet:
    def __init__(self) -> None:
        self.dataFrame: pandas.DataFrame = pandas.DataFrame()
        self.featureMatrix_train: numpy.ndarray = numpy.ndarray([])
        self.featureMatrix_test: numpy.ndarray = numpy.ndarray([])
        self.DependentVector_train: numpy.ndarray = numpy.ndarray([])
        self.DependentVector_test: numpy.ndarray = numpy.ndarray([])

    def validateDataSet(self, dataSet: pandas.DataFrame) -> bool:
        # Check if last column is binary categorical feature in values [-1, 1]
        lastFeatureValid: bool = dataSet.iloc[:,-1].isin([-1,1]).all()

        # Check if all columns except the last contain only numeric value
        remFeaturesValid: bool = True
        for col in dataSet.columns[:-1]:
            # check if current column is numeric by attempting to convert its string values to numeric values 
            try:
                dataSet[col] = pandas.to_numeric(dataSet[col], errors='raise')
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
        numericImputer  = SimpleImputer(missing_values=numpy.nan, strategy="mean")
        numericImputer.fit(self.dataFrame[self.dataFrame.columns[:-1]])
        self.dataFrame[self.dataFrame.columns[:-1]] = numericImputer.transform(
            self.dataFrame[self.dataFrame.columns[:-1]]
        )

    def test_train_splitDataSet(self) -> None:
        featureMatrix = self.dataFrame.iloc[:,:-1]
        dependentVector = self.dataFrame.iloc[:,-1]
        self.featureMatrix_train, self.featureMatrix_test, self.DependentVector_train, self.DependentVector_test = train_test_split(
            featureMatrix,
            dependentVector,
            test_size=0.2,
            random_state=0
        )

    def featureScale_trainingSet(self) -> None:
        stdScaler = StandardScaler()
        self.featureMatrix_train = stdScaler.fit_transform(self.featureMatrix_train)
        self.featureMatrix_test = stdScaler.fit_transform(self.featureMatrix_test)
    

         

        """
        numericImputer = SimpleImputer(missing_values=numpy.nan, strategy="mean")
        numericImputer.fit(
            dataSet[dataSet.columns[:-1]]
        )
        result:pandas.DataFrame = numericImputer.transform(
            dataSet[dataSet.columns[:-1]]
        )
        return result"""