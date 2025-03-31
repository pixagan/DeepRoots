##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.
import numpy as np

def train_test_split(data, labels, ratio=0.8):

    nDataPoints = len(data)

    nTrain = int(nDataPoints * ratio)

    
    all_indices = np.arange(nDataPoints)

    train_indices = np.random.choice(nTrain, size=nTrain, replace=False)

    test_indices = np.setdiff1d(all_indices, train_indices)


    train_data = data[train_indices]
    train_labels = labels[train_indices]

    test_data = data[test_indices]
    test_labels = labels[test_indices]

    return train_data, train_labels, test_data, test_labels