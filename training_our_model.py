from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from logger import logger
from training_preprocessing import get_data
from model_file_operation import file_operation


class get_model:
    def __init__(self, data):
        self.df=data
        self.rf=RandomForestRegressor()
        self.xgb=XGBRegressor()
        self.get_data=get_data()
        self.x=self.df.drop(columns="VISIBILITY")
        self.y=self.df["VISIBILITY"]
        self.train_x,self.test_x,self.train_y,self.test_y=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
        self.log=logger()
        self.model_file=file_operation()

    def get_best_param_for_random_forest(self):
        param_grid={"n_estimators": [10, 50, 100], "criterion": ['squared_error', 'absolute_error'], "max_depth":[6,8,10,12]}
        grid=GridSearchCV(estimator=self.rf, param_grid=param_grid, cv=5)
        grid.fit(self.train_x,self.train_y)
        n_estimators=grid.best_params_['n_estimators']
        criterion=grid.best_params_['criterion']
        max_depth=grid.best_params_['max_depth']
        self.rf=RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
        self.rf.fit(self.train_x,self.train_y)
        self.log.apply_log('Logs/log.txt',msg='random forest model get trained')
        return self.rf
    
    def get_best_param_for_xg_boost(self):
        self.xgb=XGBRegressor()
        self.xgb.fit(self.train_x,self.train_y)
        self.log.apply_log('Logs/log.txt',msg='xgboost regressor model get trained')
        return self.xgb


    def get_best_model(self):
         rf=self.get_best_param_for_random_forest()
         print(self.test_x)
         print(self.test_y)
         xgb=self.get_best_param_for_xg_boost()
         score_xgb=xgb.score(self.test_x,self.test_y)
         score_rf=rf.score(self.test_x,self.test_y)
         l=[score_rf,score_xgb]
         print(l)
         m=max(l)
         i=l.index(m)
         if i==0:
             self.log.apply_log('Logs/log.txt', msg='best model random forest classifier founded')
             return score_rf,rf,'random_forest'
         if i==1:
             self.log.apply_log('Logs/log.txt', msg='best model xgboost  founded')
             return score_xgb, xgb,'xgboost'




