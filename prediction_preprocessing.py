import numpy as np
import pandas as pd
from logger import logger
import os

class get_data:
    def __init__(self):
        self.data_path='Prediction_file_from_db/input_file.csv'
        self.log=logger()


    def data(self):
        try:
            df=pd.read_csv(self.data_path)
            self.log.apply_log(file_name='Logs/log.txt',msg='data get loaded from csv file to pandas dataframe')
            return df
        except Exception as e:
            self.log.apply_log(file_name='Logs/log.txt',msg='cannot load data')
            raise e

    def remove_col(self,data,col_name):
        df=data
        df1=df.drop(columns=col_name)
        self.log.apply_log(file_name='Logs/log.txt',msg='deleted {} columns'.format(col_name))
        return df1

