##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

import numpy as np

#one hot encoding for classification
def one_hot_encoding(data, classes_in):

    nDataPoints = len(data)
    nClasses    = len(classes_in)

    encoding = np.zeros([nDataPoints, nClasses])

    for i in range(0, nDataPoints):

        encoding[i, classes_in.index(data[i])] = 1.0

    return encoding