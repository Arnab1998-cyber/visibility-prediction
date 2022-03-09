from prediction_preprocessing import get_data
from model_file_operation import file_operation
from prediction_raw_data_validation import raw_data_validation
from prediction_db_operation import db_operation
from logger import logger
import os
import shutil
import pandas as pd
import json

class prediction:
    def __init__(self,prediction_files):
        self.prediction_files=prediction_files
        self.log=logger()
        self.file=file_operation()
        self.raw_data=raw_data_validation(path=self.prediction_files)
        self.db=db_operation()
        self.data=get_data()

    def get_prediction(self):
        try:
            #a,b,c,d,e=self.raw_data.values_from_schema()
            #regex=self.raw_data.manual_regex_creation()
            #self.raw_data.validation_name_raw(regex=regex,length_of_date=b,length_of_time=c)
            #self.raw_data.validation_col_length_raw(col_length=d)
            #self.raw_data.check_null_column()
            #conn=self.db.data_base_connection(database_name='prediction')
            #self.db.create_table(database_name='prediction',column_name=e)
            #self.db.preprocess_good_data()
            #self.db.insert_good_data_into_table(database='prediction')
            #self.db.upload_data_from_table_into_final_csv(database='prediction')
            df=self.data.data()
            date=df['DATE']
            df = df.drop(columns=['DATE','WETBULBTEMPF', 'DRYBULBTEMPF', 'SeaLevelPressure'])
            kmeans=self.file.load_cluster_model()
            cluster=kmeans.predict(df)
            df['cluster']=cluster
            df['DATE']=date
            l=list(df['cluster'].unique())
            l.sort()
            count=0
            if os.path.isdir('prediction_output_files'):
                shutil.rmtree('prediction_output_files')
            os.mkdir('prediction_output_files')
            for i in l:
                df2=df[df['cluster']==i]
                date_1=df2['DATE']
                df2_without_cluster_date=self.data.remove_col(data=df2,col_name=['cluster','DATE'])
                model=self.file.find_correct_model_for_cluster(cluster_number=i)[1]
                y=model.predict(df2_without_cluster_date)
                df2_without_cluster_date['DATE']=date_1
                df3=df2_without_cluster_date
                df3['VISIBILITY']=y
                new_df=df3[['DATE','VISIBILITY']]
                new_df.to_csv('Prediction_output_files/Visibility_prediction_cluster{}.csv'.format(i))
                if count==0:
                    new_df.to_csv('Prediction_output_files/Visibility_prediction.csv',mode='a+')
                    count=count+1
                else:
                    new_df.to_csv('Prediction_output_files/Visibility_prediction.csv',mode='a+',header=False)
            new_df=pd.read_csv('prediction_output_files/Visibility_prediction.csv')
            result=pd.DataFrame(new_df[['DATE','VISIBILITY']])
            self.log.apply_log('Logs/log.txt','predicted value inserted to csv file')
            outcome_path='Prediction_output_files/Visibility_prediction.csv'
            return result.head().to_json(orient='records'), outcome_path
        except Exception as e:
            raise e




