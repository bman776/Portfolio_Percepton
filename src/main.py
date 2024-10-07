from GUI import GUI

import os, sys, random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize

def main():
    gui = GUI()
    gui.run()

if __name__ == "__main__":
    main()