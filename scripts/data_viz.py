import matplotlib.pyplot as plt
from scipy.stats import norm as normal
import numpy as np
import os
import pandas as pd

def get_project_dir():
    """Returns the directory of the VS Code project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = get_project_dir()


def read_data(file_name):
    """Reads numerical data from a text file."""
    dir = os.path.join(PROJECT_DIR,'data', file_name)
    # print("Data directory:", dir)
    file_path = os.path.join(PROJECT_DIR,'data', file_name)
    data = pd.read_csv(file_path,index_col=0)
    return data


def plot_histogram(data, column, bins=30):
    """Plots a histogram of the specified column in the data."""
    data_arr = data[column].to_numpy()
    
    # Convert boolean to integer if necessary
    if data_arr.dtype == bool:
        data_arr = data_arr.astype(int)
        print("Converted boolean to integer.")

    plt.figure(figsize=(10, 6))
    plt.hist(data_arr, bins=bins, color='blue', alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_norm(data, column,ax):
    """Plots a normal probability plot of the specified column in the data."""
    data_arr = data[column].to_numpy()
    
    # Convert boolean to integer if necessary
    if data_arr.dtype == bool:
        data_arr = data_arr.astype(int)
        print("Converted boolean to integer.")

    norm_mean = np.mean(data_arr)
    norm_std = np.std(data_arr)

    print(f'Normalized Mean: {norm_mean}\nNormalized Std Dev: {norm_std}')

    domain = np.linspace(norm_mean - 3*norm_std, norm_mean + 3*norm_std, 100)

    ax.figure(figsize=(10, 6))
    p = normal.pdf(domain, loc=norm_mean, scale=norm_std)
    ax.plot(domain, p, color='red')
    ax.scatter(data_arr, np.zeros_like(data_arr), alpha=0.5)
    ax.set_title(f'Normal Probability Plot of {column}')
    ax.grid(True)
