import mysql.connector as connection
import os
import csv
import pandas as pd

from logger import logger

class db_operation:
    def __init__(self):
        self.path='Training_Database/'
        self.bad_file_path='Prediction_raw_checking_file/bad_csv_files'
        self.good_file_path='Prediction_raw_checking_file/good_csv_files'
        self.log=logger()

    def data_base_connection(self,database_name):
        try:
            conn = connection.connect(host="localhost", user="root", passwd="Arnab@kaikhali52",
                                  database=database_name,use_pure=True)
            self.log.apply_log('Logs/log.txt', msg='database conection has been established')
        except Exception as e:
            self.log.apply_log('Logs/log.txt',msg='cannection can not be established')
            raise e
        return conn

    def create_table(self,database_name, column_name):
        try:
            conn=self.data_base_connection(database_name=database_name)
            cur=conn.cursor()
            cur.execute('drop table if exists good_raw_data_visibility')
            for key in column_name:
                type=column_name[key]
                try:
                    cur.execute('create table good_raw_data_visibility ({} {}(30))'.format(key,type))
                except:
                    cur.execute('alter table good_raw_data_visibility add {} {}(10)'.format(key,type))
            conn.commit()
            conn.close()
            self.log.apply_log('Logs/log.txt', msg='table created in database')
        except Exception as e:
            self.log.apply_log('Logs/log.txt', msg='there is a problem in creating table')
            raise e


    def preprocess_good_data(self):
        try:
            good_data_file = self.good_file_path
            for file in os.listdir(good_data_file):
                df = pd.read_csv(good_data_file + '/' + file)
                df['DATE']=df["DATE"].apply(lambda x : "'"+str(x)+"'")
                df.to_csv(good_data_file + '/' + file, index=None, header=True)
            self.log.apply_log('Logs/log.txt', msg='In every file, date column preprocessed')
        except Exception as e:
            raise e

    def insert_good_data_into_table(self, database):
        conn=self.data_base_connection(database_name=database)
        cur=conn.cursor()
        good_file_path=self.good_file_path
        bad_file_path=self.bad_file_path
        l=[file for file in os.listdir(good_file_path)]
        try:
            for file in l:
                try:
                    path=good_file_path+'/'+file
                    with open(path,'r') as w:
                        m=w.readline()
                        m=m.rstrip('\n')
                    with open(path,'r') as f:
                        next(f)
                        reader=csv.reader(f,delimiter='\n')
                        for line in enumerate(reader):
                            for l in line[1]:
                                print(l)
                                cur.execute("insert into good_raw_data_visibility values({})".format(l))
                                conn.commit()
                    self.log.apply_log('Logs/log.txt',msg='{} loded to database successfully'.format(file))
                    f.close()
                except Exception as e:
                      raise e
        except Exception as e:
            conn.close()
            raise e


    def upload_data_from_table_into_final_csv(self,database):
        self.file_from_db='Prediction_file_from_db/'
        self.file_name='input_file.csv'
        try:
            conn=self.data_base_connection(database_name=database)
            cur=conn.cursor()
            cur.execute('select * from good_raw_data_visibility')
            result=cur.fetchall()
            header=[i[0] for i in cur.description]
            if not os.path.isdir(self.file_from_db):
                os.makedirs(self.file_from_db)
            csv_file=csv.writer(open(self.file_from_db+self.file_name, 'w',newline=''), delimiter=',')
            csv_file.writerow(header)
            csv_file.writerows(result)
            self.log.apply_log('Logs/log.txt',msg='csv file exported from database to master csv file')
        except Exception as e:
            print(e)
            raise e
        finally:
            conn.close()




