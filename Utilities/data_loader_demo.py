import numpy as np
import pandas as pd
import os

def load_data_csv(file_path, header=None, delimiter=','):
    """
    Load data from a CSV file using Pandas and convert to NumPy array.
    
    Parameters:
    file_path (str): Path to the CSV file.
    header (int, optional): Row number to use as the column names.
    delimiter (str, optional): Delimiter to use.
    
    Returns:
    np.ndarray: Data matrix.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path, header=header, delimiter=delimiter)
    return df.to_numpy()

def load_data_numpy(file_path, delimiter=','):
    """
    Load data from a text file using NumPy.
    
    Parameters:
    file_path (str): Path to the file.
    delimiter (str, optional): Delimiter to use.
    
    Returns:
    np.ndarray: Data matrix.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return np.genfromtxt(file_path, delimiter=delimiter)

if __name__ == "__main__":
    # Demonstration
    # Create a dummy csv file
    with open("dummy.csv", "w") as f:
        f.write("1.0,2.0,3.0\n4.0,5.0,6.0")
        
    print("Loading data using Pandas...")
    data_pd = load_data_csv("dummy.csv")
    print(data_pd)
    
    print("\nLoading data using NumPy...")
    data_np = load_data_numpy("dummy.csv")
    print(data_np)
    
    # Cleanup
    os.remove("dummy.csv")
