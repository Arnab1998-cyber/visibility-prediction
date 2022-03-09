import numpy as np
import pandas as pd
from logger import logger
import os

class get_data:
    def __init__(self):
        self.data_path='training_file_from_db/input_file.csv'
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



    def get_columns_with_zero_standered_deviation(self, data):
        df=data
        columns_to_drop=[]
        l=df.describe()
        for i in df.columns:
            if l[i]['std']==0:
                columns_to_drop.append(i)
        self.log.apply_log(file_name='Logs/log.txt',msg='got columns with zero deviation')
        return columns_to_drop







