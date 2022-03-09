from training_raw_data_validation import raw_data_validation
from training_db_operation import db_operation
from training_our_model import get_model
from training_preprocessing import get_data
from model_file_operation import file_operation
from training_data_cluster import k_means_clustering
from logger import logger
import os
import shutil

class training_pipeline:
    def __init__(self,training_files):
        self.training_files=training_files
        self.label = "VISIBILITY"
        self.log=logger()
        if os.path.isdir('Logs/'):
            shutil.rmtree('Logs')
        if os.path.isdir('models'):
            shutil.rmtree('models')
        self.log_file='Logs/log.txt'
        if not os.path.isdir('Logs/'):
            os.mkdir('Logs/')
        self.file_op=file_operation()
        self.raw_data = raw_data_validation(self.training_files)
        self.db=db_operation()
        self.data=get_data()

    def trainig_pipeline(self):
        try:
            #a,b,c,d,e=self.raw_data.values_from_schema()
            #regex=self.raw_data.manual_regex_creation()
            #self.raw_data.create_folder_good_bad_data()
            #self.raw_data.validation_name_raw(regex=regex,length_of_date=b,length_of_time=c)
            #self.raw_data.validation_col_length_raw(col_length=d)
            #self.raw_data.check_null_column()
            #conn=self.db.data_base_connection(database_name='training')
            #self.db.create_table(database_name='training',column_name=e)
            #self.db.preprocess_good_data()
            #self.db.insert_good_data_into_table(database='training')
            #self.db.upload_data_from_table_into_final_csv(database='training')
            df=self.data.data()
            df_without_label=df.drop(columns=self.label)
            label=df[self.label]
            df_without_label=df_without_label.drop(columns=['DATE','WETBULBTEMPF','DRYBULBTEMPF','SeaLevelPressure'])
            columns_to_drop=self.data.get_columns_with_zero_standered_deviation(df_without_label)
            kmeans=k_means_clustering(data=df_without_label)
            number_of_clusters=kmeans.elbow_plot()
            df_without_label_with_cluster,kmean_model=kmeans.create_clusters(no_of_cluster=number_of_clusters)
            list_of_clusters=list(df_without_label_with_cluster['cluster'].unique())
            list_of_clusters.sort()
            print(list_of_clusters)
            print(df_without_label_with_cluster)
            df_without_label_with_cluster[self.label]=label
            new_df=df_without_label_with_cluster
            for i in list_of_clusters:
                print(new_df)
                clustered_data=new_df[new_df['cluster']==i] #with cluster and label column
                clustered_dataframe=clustered_data.drop(columns='cluster')
                training=get_model(clustered_dataframe) # without cluster
                accurecy_score=training.get_best_model()[0]
                file_name=training.get_best_model()[2]
                print(accurecy_score)
                model=training.get_best_model()[1]
                self.file_op.save_model(model=model,file_name=file_name+'_{}'.format(i))
                self.log.apply_log(self.log_file,'cluster {} get trained into desired model'.format(i))
        except Exception as e:
            self.log.apply_log(self.log_file,'model training unseccessfull')
            raise e

