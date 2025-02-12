##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

#one hot encoding for classification
def one_hot_encoding(classes_in):
    
    encoding = np.zeros([len(classes_in), 6])
    for iC in range(0, len(classes_in)):
        if(classes_in[iC] == 'Spam'):
            encoding[iC, 0] = 1

        if(classes_in[iC] == 'Social Media'):
            encoding[iC, 1] = 1

        if(classes_in[iC] == 'Work'):
            encoding[iC, 2] = 1
        
        if(classes_in[iC] == 'Marketing'):
            encoding[iC, 3] = 1
        
        if(classes_in[iC] == 'Important'):
            encoding[iC, 4] = 1
        
        if(classes_in[iC] == 'Personal'):
            encoding[iC, 5] = 1

    return encoding