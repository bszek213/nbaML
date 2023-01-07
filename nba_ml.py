#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:01:02 2022

@author: brianszekely
"""
from nba_web_scraper import html_to_df_web_scrape_NBA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error #explained_variance_score
import time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor#, AdaBoostRegressor
# from scipy.stats import uniform
from os import getcwd#, mkdir
from os.path import join, exists
# from scipy import stats
import yaml
from sklearn.inspection import permutation_importance
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
# import pickle
import sys
from scipy import stats
from tqdm import tqdm
from time import sleep
# from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from boruta import BorutaPy
import joblib
"""
TODO: Scale data, right now you are not and that may be leading to overfitting issues
need to finish this, so add inverse_transform() function and scale the prediction data
"""
team_list = ['CHO','MIL','UTA','SAC','MEM','LAL',
             'MIA','IND','HOU','PHO','ATL','MIN',
             'SAS','BOS','CLE','GSW','WAS','POR',
             'LAC','NOP','DAL','BRK','NYK','ORL',
             'PHI','CHI','DEN','TOR','OKC','DET']
class nba_regressor():
    def __init__(self):
        print('initialize class nba_regressor')
        self.all_data = pd.DataFrame()
    def read_hyper_params(self):
        final_dir = join(getcwd(), 'hyper_params_regress.yaml')
        isExists = exists(final_dir)
        if isExists == True:
            with open(final_dir) as file:
                self.hyper_param_dict = yaml.load(file, Loader=yaml.FullLoader)
    def get_teams(self):
        year_list_find = []
        year_list = [2023,2022,2021,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010]
        if exists(join(getcwd(),'year_count.yaml')):
            with open(join(getcwd(),'year_count.yaml')) as file:
                year_counts = yaml.load(file, Loader=yaml.FullLoader)
        else:
            year_counts = {'year':year_list_find}
        if year_counts['year']:
            year_list_check =  year_counts['year']
            year_list_find = year_counts['year']
            year_list = [i for i in year_list if i not in year_list_check]
            print(f'Need data for year: {year_list}')
        if year_list:   
            for year in tqdm(year_list):
                # team_names = all_teams.sort_values()  
                final_list = []
                self.year_store = year
                for abv in tqdm(team_list):    
                    try:
                        print() #tqdm things
                        print(f'current team: {abv}, year: {year}')
                        # https://www.basketball-reference.com/teams/BOS/2023/gamelog/
                        basic = 'https://www.basketball-reference.com/teams/' + abv + '/' + str(self.year_store) + '/gamelog/'
                        adv = 'https://www.basketball-reference.com/teams/' + abv + '/' + str(self.year_store) + '/gamelog-advanced/'
                        df_inst = html_to_df_web_scrape_NBA(basic,adv,abv,self.year_store)
                        final_list.append(df_inst)
                    except:
                        print(f'{abv} data are not available')
                    sleep(5) #I get get banned for a small period of time if I do not do this  
                final_data = pd.concat(final_list)
                if exists(join(getcwd(),'all_data_regressor.csv')):
                    self.all_data = pd.read_csv(join(getcwd(),'all_data_regressor.csv'))  
                self.all_data = pd.concat([self.all_data, final_data.dropna()])
                if not exists(join(getcwd(),'all_data_regressor.csv')):
                    self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'))
                self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'))
                year_list_find.append(year)
                print(f'year list after loop: {year_list_find}')
                with open(join(getcwd(),'year_count.yaml'), 'w') as write_file:
                    yaml.dump(year_counts, write_file)
                    print(f'writing {year} to yaml file')
        else:
            self.all_data = pd.read_csv(join(getcwd(),'all_data_regressor.csv'))
        print('len data: ', len(self.all_data))
        self.all_data = self.all_data.drop_duplicates(keep='last')
        print(f'length of data after duplicates are dropped: {len(self.all_data)}')
    def delete_opp(self):
        """
        Drop any opponent data, as it may not be helpful when coming to prediction. Hard to estimate with running average
        """
        for col in self.all_data.columns:
            if 'opp' in col:
                self.all_data.drop(columns=col,inplace=True)
    def split(self):
        self.delete_opp()
        for col in self.all_data.columns:
            if 'Unnamed' in col:
                self.all_data.drop(columns=col,inplace=True)
        self.y = self.all_data['pts']
        self.x = self.all_data.drop(columns=['pts'])
        self.pre_process()
    def pre_process(self):   
        # Find features with correlation greater than 0.90
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # to_drop = [column for column in upper.columns if any(upper[column] >= 0.8)]
        # self.drop_cols = to_drop
        # self.drop_cols = []
        # self.drop_cols.append('game_result') #poor interpretation in regression problems
        self.x_no_corr = self.x#.drop(columns=to_drop)
        cols = self.x_no_corr.columns
        # print(f'Columns dropped: {self.drop_cols}')
        #Remove outliers with 1.5 +/- IQR
        print(f'old feature dataframe shape before outlier removal: {self.x_no_corr.shape}')
        for col_name in cols:
            Q1 = np.percentile(self.x_no_corr[col_name], 25)
            Q3 = np.percentile(self.x_no_corr[col_name], 75)
            IQR = Q3 - Q1
            upper = np.where(self.x_no_corr[col_name] >= (Q3+3.0*IQR)) #1.5 is the standard, use two to see if more data helps improve model performance
            lower = np.where(self.x_no_corr[col_name] <= (Q1-3.0*IQR)) 
            self.x_no_corr.drop(upper[0], inplace = True)
            self.x_no_corr.drop(lower[0], inplace = True)
            self.y.drop(upper[0], inplace = True)
            self.y.drop(lower[0], inplace = True)
            if 'level_0' in self.x_no_corr.columns:
                self.x_no_corr.drop(columns=['level_0'],inplace = True)
            self.x_no_corr.reset_index(inplace = True)
            self.y.reset_index(inplace = True, drop=True)
        self.x_no_corr.drop(columns=['level_0','index'],inplace = True)
        print(f'new feature dataframe shape after outlier removal: {self.x_no_corr.shape}')

        #split data into train and test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_no_corr, self.y, train_size=0.8)
        #Scale data - minmaxscaler
        self.x_scaler = MinMaxScaler()
        self.x_train_scaled = self.x_scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.x_scaler.transform(self.x_test)
        cols = self.x_train.columns.to_list()
        self.x_train_cols = self.x_train.columns.to_list()
        self.y_train_cols = self.y_train.name
        self.x_train = pd.DataFrame(self.x_train_scaled,columns=[self.x_train_cols])
        self.x_test = pd.DataFrame(self.x_test_scaled,columns=[self.x_train_cols])
        #TODO: Fix error with prob plots
        # for col_name in cols:
        #     self.prob_plots(col_name)
        #plot heat map
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(30,30))
        sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")    
        plt.tight_layout()
        plt.savefig('correlations.png',dpi=350)
        plt.close()
    def prob_plots(self,col_name):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #TODO: FIx error with prob plots
        prob = stats.probplot(self.x_train[col_name], dist=stats.norm, plot=ax1)
        title = f'probPlot of training data against normal distribution, feature: {col_name}'  
        ax1.set_title(title,fontsize=10)
        save_name = 'probplot_' + col_name + '.png'
        plt.tight_layout()
        plt.savefig(join(getcwd(), 'prob_plots_regress',save_name), dpi=300)
        plt.close()
    def boruta_feature_selection(self):
        RandForclass = RandomForestRegressor()
        feat_selector = BorutaPy(
                verbose=2,
                estimator=RandForclass,
                n_estimators='auto',
                max_iter=10)  # number of iterations to perform
        feat_selector.fit(np.array(self.x_train),np.array(self.y_train))
        print(feat_selector.support_)
        print("Ranking and support for all features")
        self.drop_cols_boruta = []
        for i in range(len(feat_selector.support_)):
            if feat_selector.support_[i]:
                print(f'Save feature: {self.x_train.columns[i][0]}')
            else:
                print(f'Drop feature: {self.x_train.columns[i][0]}')
                self.drop_cols_boruta.append(self.x_train.columns[i][0])
        self.drop_cols_boruta.append('game_result')
        print(f'Features to drop based on Boruta algorithm: {self.drop_cols_boruta}')
        self.x_train.drop(columns=self.drop_cols_boruta, inplace=True)
        self.x_test.drop(columns=self.drop_cols_boruta, inplace=True)
        # RandForclass.fit(self.x_train,self.y_train)
    def machine(self):
        if sys.argv[1] == 'tune':
            #RANDOM FOREST
            #Drop columns with Boruta Algorithm
            self.boruta_feature_selection()
            RandForclass = RandomForestRegressor()
            rows, cols = self.x_train.shape
            Rand_perm = {
                'criterion' : ["squared_error", "poisson"], #absolute_error - takes forever to run
                'n_estimators': range(300,500,100),
                # 'min_samples_split': np.arange(2, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2'],
                'max_depth': np.arange(2,cols,1),
                'min_samples_leaf': np.arange(1,3,1)
                }
            #['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'matthews_corrcoef', 'max_error', 'mutual_info_score', 'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']
            clf_rand = GridSearchCV(RandForclass, Rand_perm, 
                                scoring=['neg_root_mean_squared_error','explained_variance'],
                                cv=10,refit='neg_root_mean_squared_error',verbose=3, n_jobs=-1)
            search_rand = clf_rand.fit(self.x_train,self.y_train)
            joblib.dump(search_rand, "./randomForestModelTuned.joblib", compress=9)
            print('RandomForestRegressor - best params: ',search_rand.best_params_)
            #MULTI-LAYER PERCEPTRON
            # MLPClass = MLPRegressor()
            # MLP_perm = {
            #     'activation':['identity','relu','tanh'],
            #     'solver' : ['lbfgs', 'sgd', 'adam'],
            #     'learning_rate' : ['constant', 'invscaling', 'adaptive'],
            #     'learning_rate_init' : np.arange(0.001, 0.01, 0.01, dtype=float),
            #     'max_iter': range(100,1000,200),
            #     # 'tol': np.arange(0.001, 0.005, 0.001, dtype=float)
            #     }
            # clf_MLP = GridSearchCV(MLPClass, MLP_perm, scoring=['neg_root_mean_squared_error'],
            #                    refit='neg_root_mean_squared_error', verbose=4, n_jobs=-1)
            # search_MLP= clf_MLP.fit(self.x_train,self.y_train)
            # print('MultiLayerPerceptron - best params: ',search_MLP.best_params_)
            return search_rand
        else:
            print('load tuned model')
            #RANDOM FOREST
            RandForclass=joblib.load("./randomForestModelTuned.joblib")
            print(f'Current RandomForestRegressor Parameters: {RandForclass.best_params_}')
            # RandForclass = RandomForestRegressor(
            #     criterion='squared_error',
            #     max_features='sqrt', 
            #     min_samples_split=3, 
            #     n_estimators=406
            #     )#.fit(self.x_train,self.y_train)
            
            RAND_rmse = mean_squared_error(self.y_test, RandForclass.predict(self.x_test),squared=False)
            #MULTILAYER PERCEPTRON
            # MLPClass = MLPRegressor(
            #     activation='identity',
            #     learning_rate='adaptive',
            #     learning_rate_init=0.004,
            #     max_iter=900,
            #     solver='lbfgs'
            #     )
            # ADA_with_MLP = AdaBoostRegressor(base_estimator=MLPClass,random_state=0, n_estimators=100).fit(self.x_train,self.y_train)
            # MLP_err = r2_score(self.y_test, ADA_with_MLP.predict(self.x_test))
            # MLP_rmse = np.sqrt(mean_squared_error(self.y_test, ADA_with_MLP.predict(self.x_test)))
            #LINEAR REGRESSION
            LinReg = LinearRegression()
            linClass = LinReg.fit(self.x_train,self.y_train)
            Line_rmse = mean_squared_error(self.y_test, linClass.predict(self.x_test),squared=False)
            # print('Adaboost with MultiLayerPerceptron rmse',MLP_rmse)
            # print('Adaboost with MultiLayerPerceptron accuracy',MLP_err)
            print('RandomForestRegressor rmse',RAND_rmse)
            print('LinearRegression rmse',Line_rmse)
            return RandForclass
    def predict_two_teams(self,model):
        while True:
                print(f'list of teams: {sorted(team_list)}')
                try:
                    team_1 = input('team_1: ')
                    if team_1 == 'exit':
                        break
                    team_2 = input('team_2: ')
                    # print(f'is {team_1} home or away:')
                    # team_1_loc = input('type home or away: ')
                    # year = int(input('year: '))
                    #Only use the previous year when it is early season
                    # year = 2022
                    # #2021
                    # team_1_url = 'https://www.basketball-reference.com/teams/' + team_1.upper() + '/' + str(year) + '/gamelog/'
                    # team_1_url_adv = 'https://www.basketball-reference.com/teams/' + team_1.upper() + '/' + str(year) + '/gamelog-advanced/'
                    # team_2_url = 'https://www.basketball-reference.com/teams/' + team_2.upper() + '/' + str(year) + '/gamelog/'
                    # team_2_url_adv = 'https://www.basketball-reference.com/teams/' + team_2.upper() + '/' + str(year) + '/gamelog-advanced/'
                    # team_1_df2022 = html_to_df_web_scrape_NBA(team_1_url,team_1_url_adv,team_1,year)
                    # team_2_df2022 = html_to_df_web_scrape_NBA(team_2_url,team_2_url_adv,team_2,year)
                    #2022
                    year = 2023
                    team_1_url = 'https://www.basketball-reference.com/teams/' + team_1.upper() + '/' + str(year) + '/gamelog/'
                    team_1_url_adv = 'https://www.basketball-reference.com/teams/' + team_1.upper() + '/' + str(year) + '/gamelog-advanced/'
                    team_2_url = 'https://www.basketball-reference.com/teams/' + team_2.upper() + '/' + str(year) + '/gamelog/'
                    team_2_url_adv = 'https://www.basketball-reference.com/teams/' + team_2.upper() + '/' + str(year) + '/gamelog-advanced/'
                    team_1_df2023= html_to_df_web_scrape_NBA(team_1_url,team_1_url_adv,team_1,year)
                    team_2_df2023 = html_to_df_web_scrape_NBA(team_2_url,team_2_url_adv,team_2,year)
                    #concatenate 2022 and 2023
                    # final_data_1 = pd.concat([team_1_df2022, team_1_df2023])
                    # final_data_2 = pd.concat([team_2_df2022, team_2_df2023])
                    final_data_1 = team_1_df2023
                    final_data_2 = team_2_df2023
                    #clean team 1 labels
                    # team_1_df['game_result'] = team_1_df['game_result'].str.replace('W','')
                    # team_1_df['game_result'] = team_1_df['game_result'].str.replace('L','')
                    # team_1_df['game_result'] = team_1_df['game_result'].str.replace('(','')
                    # team_1_df['game_result'] = team_1_df['game_result'].str.replace(')','')
                    # team_1_df['game_result'] = team_1_df['game_result'].str.split('-').str[0]
                    # team_1_df['game_result'] = team_1_df['game_result'].str.replace('-','')
                    # final_data_1 = team_1_df.replace(r'^\s*$', np.NaN, regex=True)
                    # #clean team 2 labels
                    # team_2_df['game_result'] = team_2_df['game_result'].str.replace('W','')
                    # team_2_df['game_result'] = team_2_df['game_result'].str.replace('L','')
                    # team_2_df['game_result'] = team_2_df['game_result'].str.replace('(','')
                    # team_2_df['game_result'] = team_2_df['game_result'].str.replace(')','')
                    # team_2_df['game_result'] = team_2_df['game_result'].str.split('-').str[0]
                    # team_2_df['game_result'] = team_2_df['game_result'].str.replace('-','')
                    # final_data_2 = team_2_df.replace(r'^\s*$', np.NaN, regex=True) #replace empty string with NAN
                    for col in final_data_1.columns:
                        if 'Unnamed' in col:
                            final_data_1.drop(columns=col,inplace=True)
                    for col in final_data_2.columns:
                        if 'Unnamed' in col:
                            final_data_2.drop(columns=col,inplace=True)
                    # if 'Unnamed: 0' in final_data_1.columns:
                    #     final_data_1 = final_data_1.drop(columns=['Unnamed: 0'])
                    # if 'Unnamed: 0' in final_data_2.columns:
                    #     final_data_2 = final_data_2.drop(columns=['Unnamed: 0'])
                    
                    #drop cols
                    final_data_1.drop(columns=self.drop_cols_boruta, inplace=True)
                    final_data_2.drop(columns=self.drop_cols_boruta, inplace=True)
                    final_data_1.drop(columns=['pts'], inplace=True)
                    final_data_2.drop(columns=['pts'], inplace=True)
                    #dropnans
                    final_data_1.dropna(inplace=True)
                    final_data_2.dropna(inplace=True)
                    #minMaxScale data
                    inst = MinMaxScaler()
                    cols = final_data_1.columns.to_list()
                    final_data_1 = inst.fit_transform(final_data_1)
                    final_data_2 = inst.transform(final_data_2)
                    final_data_1 = pd.DataFrame(final_data_1,columns=[cols])
                    final_data_2 = pd.DataFrame(final_data_2,columns=[cols])
                    #create data for prediction
                    # df_features_1 = final_data_1.dropna().median(axis=0,skipna=True).to_frame().T
                    # df_features_2 = final_data_2.dropna().median(axis=0,skipna=True).to_frame().T
                    #Predicions
                    data1 = final_data_1.dropna().median(axis=0,skipna=True).to_frame().T
                    data2 = final_data_2.dropna().median(axis=0,skipna=True).to_frame().T
                    if not data1.isnull().values.any() and not data2.isnull().values.any():
                        #Try to find the moving averages that work
                        ma_range = np.arange(2,31,1)
                        #USE THIS AFTER GAMES ARE OVER TO FIGURE OUT WHICH RANGES ARE BEST
                        # team_won = input('who won: ')
                        # correct_ranges = []
                        # for ma in tqdm(ma_range):
                        #     data1 = final_data_1.dropna().rolling(ma).mean()
                        #     data2 = final_data_2.dropna().rolling(ma).mean()
                        #     team_1_predict = model.predict(data1.iloc[-1:])
                        #     team_2_predict = model.predict(data2.iloc[-1:])
                        #     if team_1_predict < team_2_predict:
                        #         if team_won == team_2:
                        #             correct_ranges.append(ma)
                        #             print(f'{ma} is correct')
                        #     if team_1_predict > team_2_predict:
                        #         if team_won == team_1:
                        #             correct_ranges.append(ma)
                        #             print(f'{ma} is correct')
                        # print(f'correct ranges: {correct_ranges}')
                        # plt.figure()
                        team_1_count = 0
                        team_1_ma = []
                        team_2_count = 0
                        team_2_ma = []
                        for ma in tqdm(ma_range):
                            data1 = final_data_1.dropna().rolling(ma).median()
                            data2 = final_data_2.dropna().rolling(ma).median()
                            team_1_predict = model.predict(data1.iloc[-1:])
                            team_2_predict = model.predict(data2.iloc[-1:])
                            if team_1_predict > team_2_predict:
                                team_1_count += 1
                                team_1_ma.append(ma)
                            if team_1_predict < team_2_predict:
                                team_2_count += 1
                                team_2_ma.append(ma)
                        # short = 2
                        # medium = 12
                        # long = 20
                        # data1_long = final_data_1.dropna().rolling(long).mean() #long
                        # # plt.plot(data1_long['pace'].values)
                        # data2_long = final_data_2.dropna().rolling(long).mean()
                        # data1_long = data1_long.iloc[-1:]
                        # # print(data1_long['off_rtg'].values)
                        # data2_long = data2_long.iloc[-1:]
                        # data1_short = final_data_1.dropna().rolling(short).mean() #short
                        # # plt.plot(data1_short['pace'].values)
                        # data2_short= final_data_2.dropna().rolling(short).mean()
                        # data1_short = data1_short.iloc[-1:]
                        # # print(data1_short['off_rtg'].values)
                        # data2_short = data2_short.iloc[-1:]
                        # data1_med = final_data_1.dropna().rolling(medium).mean() #medium
                        # # plt.plot(data1_med['pace'].values)
                        # # plt.legend(['Long','Short','Medium'])
                        # data2_med = final_data_2.dropna().rolling(medium).mean()
                        # data1_med = data1_med.iloc[-1:]
                        # # print(data1_med['off_rtg'].values)
                        # data2_med = data2_med.iloc[-1:]
                        # #Predictions based on running means
                        # team_1_data_long_avg = model.predict(data1_long)
                        # team_2_data_long_avg = model.predict(data2_long)
                        # team_1_data_short_avg = model.predict(data1_short)
                        # team_2_data_short_avg = model.predict(data2_short)
                        # team_1_data_med_avg = model.predict(data1_med)
                        # team_2_data_med_avg = model.predict(data2_med)
                        # vote_running_avg = []
                        # if team_1_data_short_avg[0] > team_2_data_short_avg[0]:
                        #     vote_running_avg.append(team_1)
                        # else:
                        #     vote_running_avg.append(team_2)
                        # if team_1_data_med_avg[0] > team_2_data_med_avg[0]:
                        #     vote_running_avg.append(team_1)
                        # else:
                        #     vote_running_avg.append(team_2)
                        # if team_1_data_long_avg[0] > team_2_data_long_avg[0]:
                        #     vote_running_avg.append(team_1)
                        # else:
                        #     vote_running_avg.append(team_2)
                        print('===============================================================')
                        # print(f'all predictions from both models with a short ({short} game rolling mean), medium ({medium} game rolling mean), and long running average ({long} game rolling mean)')
                        # print(vote_running_avg)
                        print(f'Rolling averages with a rolling average from 2-30 games')
                        print(f'{team_1}: {team_1_count} | {team_1_ma}')
                        print(f'{team_2}: {team_2_count} | {team_2_ma}')
                        print('===============================================================')
                        #PLOT PCA FOR VISUALIZATION
                        pca_1 = PCA(n_components=2).fit(final_data_1)
                        plot_data1 = pca_1.transform(final_data_1)
                        print(pca_1.explained_variance_ratio_)
                        pca_2 = PCA(n_components=2).fit(final_data_2)
                        plot_data2 = pca_2.transform(final_data_2)
                        print(pca_2.explained_variance_ratio_)
                        # fig = plt.figure()
                        # ax = plt.axes(projection='3d')
                        # ax.plot3D(np.arange(0,len(plot_data1[:,0]),1),plot_data1[:,0], plot_data1[:,1], 'green',label=team_1)
                        # ax.plot3D(np.arange(0,len(plot_data2[:,0]),1),plot_data2[:,0], plot_data2[:,1], 'blue',label=team_2)
                        # plt.legend()
                        # plt.show()
                except Exception as e:
                    print(f'Team not found: {e}')
                #OLD CODE 
                # if not data1.isnull().values.any() and not data1.isnull().values.any():
                #     team_1_data_all = model.predict(data1)
                #     team_2_data_all = model.predict(data2)
                #     if team_1_data_all[0] > team_2_data_all[0]:
                #         team_1_total += 1
                #         game_won_team_1.append('season')
                #     else:
                #         team_2_total += 1
                #         game_won_team_2.append('season')
                #     print(f'Score prediction for {team_1} across 2022 and 2023 season: {team_1_data_all[0]} points')
                #     print(f'Score prediction for {team_2} across 2022 and 2023 season: {team_2_data_all[0]} points')
                #     print('====')
                # data1 = final_data_1.iloc[-1:].dropna().median(axis=0,skipna=True).to_frame().T
                # data2 = final_data_2.iloc[-1:].dropna().median(axis=0,skipna=True).to_frame().T
                # if not data1.isnull().values.any() and not data1.isnull().values.any():
                #     team_1_data_last = model.predict(data1)
                #     team_2_data_last = model.predict(data2)
                #     if team_1_data_last[0] > team_2_data_last[0]:
                #         team_1_total += 1
                #         game_won_team_1.append('last_game')
                #     else:
                #         team_2_total += 1
                #         game_won_team_2.append('last_game')
                #     print(f'Score prediction for {team_1} last game: {team_1_data_last[0]} points')
                #     print(f'Score prediction for {team_2} last game: {team_2_data_last[0]} points')
                #     print('====')
                # data1 = final_data_1.iloc[-3:].dropna().median(axis=0,skipna=True).to_frame().T
                # data2 = final_data_2.iloc[-3:].dropna().median(axis=0,skipna=True).to_frame().T
                # if not data1.isnull().values.any() and not data1.isnull().values.any():
                #     team_1_data_last = model.predict(data1)
                #     team_2_data_last = model.predict(data2)
                #     if team_1_data_last[0] > team_2_data_last[0]:
                #         team_1_total += 1
                #         game_won_team_1.append('last_3_game')
                #     else:
                #         team_2_total += 1
                #         game_won_team_2.append('last_3_game')
                #     print(f'Score prediction for {team_1} last 3 games: {team_1_data_last[0]} points')
                #     print(f'Score prediction for {team_2} last 3 games: {team_2_data_last[0]} points')
                #     print('====')
                # data1 = final_data_1.iloc[-5:].dropna().median(axis=0,skipna=True).to_frame().T
                # data2 = final_data_2.iloc[-5:].dropna().median(axis=0,skipna=True).to_frame().T
                # if not data1.isnull().values.any() and not data1.isnull().values.any():
                #     team_1_data_last2 = model.predict(data1)
                #     team_2_data_last2 = model.predict(data2)
                #     if team_1_data_last2[0] > team_2_data_last2[0]:
                #         team_1_total += 1
                #         game_won_team_1.append('last_5_games')
                #     else:
                #         team_2_total += 1
                #         game_won_team_2.append('last_5_games')
                #     print(f'Score prediction for {team_1} last 5 game: {team_1_data_last2[0]} points')
                #     print(f'Score prediction for {team_2} last 5 game: {team_2_data_last2[0]} points')
                #     print('====')
                # data1 = final_data_1.iloc[-10:].dropna().median(axis=0,skipna=True).to_frame().T
                # data2 = final_data_2.iloc[-10:].dropna().median(axis=0,skipna=True).to_frame().T
                # if not data1.isnull().values.any() and not data1.isnull().values.any():
                #     team_1_data_last3 = model.predict(data1)
                #     team_2_data_last3 = model.predict(data2)
                #     if team_1_data_last3[0] > team_2_data_last3[0]:
                #         team_1_total += 1
                #         game_won_team_1.append('last_10_games')
                #     else:
                #         team_2_total += 1
                #         game_won_team_2.append('last_10_games')
                #     print(f'Score prediction for {team_1} last 10 game: {team_1_data_last3[0]} points')
                #     print(f'Score prediction for {team_2} last 10 game: {team_2_data_last3[0]} points')
                #     print('====')
                # data1 = final_data_1.iloc[-15:].dropna().median(axis=0,skipna=True).to_frame().T
                # data2 = final_data_2.iloc[-15:].dropna().median(axis=0,skipna=True).to_frame().T
                # if not data1.isnull().values.any() and not data1.isnull().values.any():
                #     team_1_data_last5 = model.predict(data1)
                #     team_2_data_last5 = model.predict(data2)
                #     if team_1_data_last5[0] > team_2_data_last5[0]:
                #         team_1_total += 1
                #         game_won_team_1.append('last_15_games')
                #     else:
                #         team_2_total += 1
                #         game_won_team_2.append('last_15_games')
                #     print(f'Score prediction for {team_1} last 15 game: {team_1_data_last5[0]} points')
                #     print(f'Score prediction for {team_2} last 15 game: {team_2_data_last5[0]} points')
                # data1 = final_data_1.iloc[-20:].dropna().median(axis=0,skipna=True).to_frame().T
                # data2 = final_data_2.iloc[-20:].dropna().median(axis=0,skipna=True).to_frame().T
                # if not data1.isnull().values.any() and not data1.isnull().values.any():
                #     team_1_data_last5 = model.predict(data1)
                #     team_2_data_last5 = model.predict(data2)
                #     if team_1_data_last5[0] > team_2_data_last5[0]:
                #         team_1_total += 1
                #         game_won_team_1.append('last_20_games')
                #     else:
                #         team_2_total += 1
                #         game_won_team_2.append('last_20_games')
                #     print(f'Score prediction for {team_1} last 20 game: {team_1_data_last5[0]} points')
                #     print(f'Score prediction for {team_2} last 20 game: {team_2_data_last5[0]} points')
    def feature_importances(self,model):
        # model.best_estimator_.feature_importances_?
        if model != "no model":
            if 'keras' in str(model):
                imps = PermutationImportance(model,random_state=1).fit(self.x_test, self.y_test)
                print(show_weights(imps,feature_names=self.x_test.columns))
            else:
                imps = permutation_importance(model, self.x_test, self.y_test)
            if 'MLPClassifier' or 'LinearRegression' or 'PassiveAggressive' or 'keras' in str(model):
                feature_imp = pd.Series(imps.importances_mean,index=self.x_test.columns).sort_values(ascending=False)
                plt.close()
                plt.figure()
                sns.barplot(x=feature_imp,y=feature_imp.index)
                plt.xlabel('Feature Importance')
                plt.ylabel('Features')
                title_name = f'FeatureImportance - {str(model)}'
                plt.title(title_name,fontdict={'fontsize': 6})
                save_name = 'FeatureImportance' + '.png'
                plt.tight_layout()
                plt.savefig(join(getcwd(), save_name), dpi=300)
            else:
                feature_imp = pd.Series(model.best_estimator_.feature_importances_,index=self.x_test.columns).sort_values(ascending=False)
                plt.close()
                plt.figure()
                sns.barplot(x=feature_imp,y=feature_imp.index)
                plt.xlabel('Feature Importance')
                plt.ylabel('Features')
                title_name = f'FeatureImportance - {str(model)}'
                plt.title(title_name,fontdict={'fontsize': 6})
                save_name = 'FeatureImportanceRegress' + '.png'
                plt.tight_layout()
                plt.savefig(join(getcwd(), save_name), dpi=300)
def main():
    start_time = time.time()
    class_inst = nba_regressor()
    class_inst.get_teams()
    # class_inst.read_hyper_params()
    class_inst.split()
    model = class_inst.machine()
    # if not sys.argv[1] == 'tune':
    class_inst.predict_two_teams(model)
    class_inst.feature_importances(model)
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    main()
