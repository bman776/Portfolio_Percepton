import tkinter
from tkinter import ttk
from tkinter import filedialog
import tkinter.messagebox

import pathlib
import pandas 

import csv


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class GUI:
    def __init__(self) -> None:

        # define Empty data set to store input dataset
        self.dataSet: pandas.DataFrame = pandas.DataFrame()

        # define GUI
        # Create main window
        self.rootWindow = tkinter.Tk()
        self.rootWindow.title("Multiple Linear Regression Model Builder")
        self.rootWindow.geometry('500x500')
        self.rootWindow.columnconfigure(0, minsize=100, weight=1)
        self.rootWindow.columnconfigure(1, minsize=100, weight=1)
        self.rootWindow.rowconfigure(0, minsize=100, weight=4)
        self.rootWindow.rowconfigure(1, minsize=100, weight=1)
        # define data frame
        self.data_frame = tkinter.Frame(self.rootWindow, borderwidth=5, relief=tkinter.RIDGE)
        self.data_frame.pack(fill="both", expand=True, padx=10, pady=10)
        # Define load dataset button
        self.openDataSet_button = tkinter.Button(self.data_frame, text="Load Data Set", command=self.loadDataSet_csv)
        self.openDataSet_button.pack(side=tkinter.TOP, padx=10, pady=10)
        # define data tree to store and view loaded data set
        self.data_tree = ttk.Treeview(self.data_frame, show="headings")
        self.data_tree.pack(side=tkinter.TOP, padx=10, pady=10, fill='both', expand=True)
        # Define load data prompt label
        self.status_label = tkinter.Label(self.data_frame, text="Please Load \n a Data Set", padx=20, pady=10)
        self.status_label.pack(side=tkinter.TOP, padx=10, pady=10, fill='both')
        # Define build model button
        self.buildModel_button = tkinter.Button(self.data_frame, text="Build Model", command=self.buildModel)
        self.buildModel_button.pack(side=tkinter.BOTTOM, padx=10, pady=10)

    def run(self):
        self.rootWindow.mainloop()


    # DEV NOTE:
    # This function both validates the data set and ensures its valid values are converted to numeric data type
    def validateDataSet(self, dataSet: pandas.DataFrame) -> bool:
        # Check if last column is binary categorical feature in values [-1, 1]
        lastFeatureValid: bool = dataSet.iloc[:,-1].isin([-1,1]).all()

        # Check if all columns except the last contain only numeric value
        remFeaturesValid: bool = True
        for col in dataSet.columns[:-1]:
            # check if current column is numeric by attempting to convert its non-NaN string 
            # values to numeric values 
            try:
                # select non-Nan values from the current column
                col_nonNaNVals = dataSet[col][dataSet[col].notna()]
                # attempt conversion
                pandas.to_numeric(col_nonNaNVals, errors='raise')
            except ValueError as e:
                remFeaturesValid = False
                break

            # column has valid format (valid values)
            dataSet[col] = pandas.to_numeric(dataSet[col], errors='coerce')

        if not (lastFeatureValid and remFeaturesValid):
            return False
        
        return True
        

    def buildModel(self) -> None:
        pass
            
        
    def loadDataSet_csv(self):
        filePath = filedialog.askopenfilename(title="Open Data Set", filetypes=[("CSV files", "*.csv")])
        if filePath:
            # Get Input Data
            dataSet:pandas.DataFrame = pandas.read_csv(
                filePath, 
                na_values=["", " ", "\t", "NULL", "NaN", "n/a", "N/A", "-", "*", "?"], 
                skipinitialspace=True
            )
            # DEV NOTE: pandas.read_csv() fills in missing data values w/ NaN

            # Validate Input Data
            dataValid: bool = self.validateDataSet(dataSet)
            if dataValid:
                
                # Store Input Data
                self.dataSet = dataSet

                # Load data set into GUI

                # Clear previous data from GUI
                self.data_tree.delete(*self.data_tree.get_children())

                # Load current data  
                header = list(dataSet.columns)
                self.data_tree["columns"] = header
                for col in header:
                    self.data_tree.heading(col, text=col)
                    self.data_tree.column(col, width=100)

                for index, row in dataSet.iterrows():
                    self.data_tree.insert("", "end", values=list(row))

                self.status_label.config(text=f"CSV file loaded: {filePath}")

            else:
                # Display warning message to user if data set is invalid
                tkinter.messagebox.showwarning(
                    title="Invalid Data", 
                    message="The Data Set you selected does not satisfy \n the conditions for this program"
                )


