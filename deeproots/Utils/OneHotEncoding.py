##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

#one hot encoding for classification
def one_hot_encoding(classes_in, data):

    nDataPoints = len(data)
    nClasses    = len(classes_in)
    
    encoding = np.zeros([nDataPoints, nClasses])


    for i in range(0, nDataPoints):

        encoding[i, data[i]] = 1.0

    return encoding