##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

import pandas as pd


class CSVLoader:

    def __init__(self):
        print("Initializing CSV Loader")

    def load(self, filename):

        df = pd.read_csv(filename)

        return df



#----------------------------------------------



def main():

    filename = "../../datasets/carprice_data.csv"
    D1 = CSVLoader().load(filename) 
    print("Dataframe : ", D1)




#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------

        
