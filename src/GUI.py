from Data import DataSet
from Model import Perceptron

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
        self.dataSet: DataSet = DataSet()

        # define Model object
        self.perceptron: Perceptron = Perceptron()

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


        
    def loadDataSet_csv(self):
        filePath = filedialog.askopenfilename(title="Open Data Set", filetypes=[("CSV files", "*.csv")])
        if filePath:
            # Get Input Data
            self.dataSet.loadData(
                pandas.read_csv(
                    filePath, 
                    na_values=["", " ", "\t", "NULL", "NaN", "n/a", "N/A", "-", "*", "?"], 
                    skipinitialspace=True
                )
            )
            # DEV NOTE: pandas.read_csv() fills in missing data values w/ NaN

            # Validate Input Data
            dataValid: bool = self.dataSet.validateDataSet()

            if dataValid:
                # Load data set into GUI

                # Clear previous data from GUI
                self.data_tree.delete(*self.data_tree.get_children())

                # Load current data  
                header = list(self.dataSet.dataFrame.columns)
                self.data_tree["columns"] = header
                for col in header:
                    self.data_tree.heading(col, text=col)
                    self.data_tree.column(col, width=100)

                for index, row in self.dataSet.dataFrame.iterrows():
                    self.data_tree.insert("", "end", values=list(row))

                self.status_label.config(text=f"CSV file loaded: {filePath}")

            else:
                # Display warning message to user if data set is invalid
                tkinter.messagebox.showwarning(
                    title="Invalid Data", 
                    message="The Data Set you selected does not satisfy \n the conditions for this program"
                )



    def buildModel(self) -> None:
        self.dataSet.preprocessDataSet()

        # DEV NOTE: may change name of loadDataSet function for perceptron class
        # only reason to give data to model object is so it can actually build model
        
        self.perceptron.loadDataSet(self.dataSet)
        self.perceptron.executeLearningAlgorithm()

    def displayModel(self) -> None:
        
