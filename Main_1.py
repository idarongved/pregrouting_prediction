# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:32:37 2022

@author: IRo
"""
###############################################################################
# imports
import joblib
import numpy as np
import optuna
from os import listdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
# from sklearn.metrics import max_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.inspection import feature_importances_
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from library_1 import plotter, utilities

# TODO general: go through code and add comments and explenations where
# necessary and clean it up a bit

# TODO make a Github account and check it out if you do not have it yet


###############################################################################
# static variables

FOLDER = r'00_raw_data\UDK01_grouting_water'
# list of all possible inputs
INPUT = ['Prev. grouting time', 'Prev. grout take', 'Stop pressure',
         'Number of holes', 'Cement_type',
         'PenetrNormMean', 'PenetrRMSMean', 'RotaPressNormMean',
         'TerrainHeight', 'RotaPressRMSMean', 'FeedPressNormMean',
         'HammerPressNormMean', 'WaterFlowNormMean', 'WaterFlowRMSMean',
         'Q', 'ContourWidth']
# 'RQD', 'Jr', 'Jw', 'SRF', 'Jn', 'JnMult', 'Ja',

MAPPING_FEATURES = ['PegEnd', 'RockClass', 'Rock', 'Q', 'RQD', 'Jr', 'Jw',
                    'Jn', 'JnMult', 'Ja', 'SRF', 'TerrainHeight',
                    'ContourWidth']
OUTPUT = 'Grouting time'  # 'Grouting time', 'Total grout take'

OPTIMIZATION_TARGET = ['Stop pressure']
# 'Prev. grouting time', 'Prev. grout take', 'Stop pressure', 'Number of holes'


TRAIN_TEST_SPLIT = 0.25  # amount of datapoints that will be used for testing
TOLERANCE = 2  # [m]  merging tolerance

DICTIONARY = {'Antall borehull': 70, 'Injeksjonstid [time]': 35,
              'Sement mengde [kg]': 50000, 'WaterFlowNormMean': 50,
              'WaterFlowRMSMean': 12, 'WaterFlowNormMean1': -50,
              'Forrige injeksjonstid': 35, 'Forrige sementmengde': 50000}


STUDY_NAME = '2023_05_12_RF_5'  # Make filename to save study

OPTIMIZATION = True  # False, True

PAIRPLOT = False  # False, True
HEATMAP = False  # False, True

TO_PAIRPLOT = []
#  ['Prev. grouting time', 'Prev. grout take', 'Stop pressure',
#          'Number of holes', 'Cement_type',
#          'PenetrNormMean', 'PenetrRMSMean', 'RotaPressNormMean',
#          'TerrainHeight', 'RotaPressRMSMean', 'FeedPressNormMean',
#          'HammerPressNormMean', 'WaterFlowNormMean', 'WaterFlowRMSMean',
#          'Q', 'ContourWidth']

###############################################################################
# dynamic variables

if '_DT_' in STUDY_NAME:
    MODEL = 'DecisionTree'
elif '_ET_' in STUDY_NAME:
    MODEL = 'ExtraTrees'
elif '_RF_' in STUDY_NAME:
    MODEL = 'RandomForest'
else:
    raise ValueError('wrong study name specification!!')

files = listdir(FOLDER)
# fix random seed for reproducibility
np.random.seed(42)

# instantiations
pltr = plotter(INPUT, OUTPUT)
utils = utilities()

###############################################################################
# conventional data preprocessing

df = utils.merge_inj(files, folder=FOLDER)

# Read csv file with MWD
df_mwd = pd.read_csv(r'00_raw_data\UDK01_mwd_Q_rocktype - Copy.csv', sep=';',
                     decimal='.')

# merge mapping data from last blasting to new fan
df_mwd.sort_values('PegStart', inplace=True)

# df_mwd_mapping = df_mwd[MAPPING_FEATURES]
# df_mwd.drop(MAPPING_FEATURES, axis=1, inplace=True)


df = pd.merge_asof(left=df.sort_values('Pel'),
                   right=df_mwd.sort_values('PegEnd'),
                   left_on='Pel', right_on='PegEnd',
                   direction='nearest', tolerance=TOLERANCE)

# change Q-values to log(Q)
df['Q'] = np.log10(df['Q'])
    
# remove columns we dont need
to_delete = []

for f in df:
    if 'Deviation' in f or 'Variance' in f or 'Kurtosis' in f or 'Skewness' in f:
        to_delete.append(f)

df.drop(to_delete, axis=1, inplace=True)

# drope the rows without numbers
df.dropna(inplace=True)

# Create a new df for prev fan output
df_prevfan = pd.DataFrame()

df_prevfan['Forrige injeksjonstid'] = df['Injeksjonstid [time]']
df_prevfan['Forrige sementmengde'] = df['Sement mengde [kg]']

# Shift all the values in df_prevfan on index down
df_prevfan = df_prevfan.shift(1)

# Drop the last row since it's the last fan
last_row = len(df_prevfan)-1
df_prevfan.drop(df_prevfan.index[last_row], inplace=True)

# fix the index of df_prevfan and save
df_prevfan.index = np.arange(len(df_prevfan))
df_prevfan.to_excel(r'01_processed_data\prev_fan.xlsx')


# fix the index of df
df.index = np.arange(len(df))

# merg df and df_prevfan by using index
df = pd.merge(df, df_prevfan, left_on=df.index, right_on=df_prevfan.index)

# drope the rows without numbers
df.dropna(inplace=True)

# save total data in excel file
df.to_excel(r'01_processed_data\total_data_file4.xlsx')


##############################################################################
# ML data preprocessing

df = utils.remove_outliers(data=df, dictionary=DICTIONARY,
                           before_after_plot=False)

# make all features numbers. Mikrosement = 1 and Industrisement = 0
df['Sementtype'] = np.where(df['Sementtype'] == 'Mikrosement', 1, 0)

# get all possible input feature combinations
input_ = utils.all_combinations(INPUT[3:], min_features=4)
input_ = [INPUT[:3] + list(comb) for comb in input_]
print(f'There are {len(input_)} possible input feature combinations\n')

# make all norwegian feature names english
df = df.rename(columns={'Antall borehull': 'Number of holes',
                        'Injeksjonstid [time]': 'Grouting time',
                        'Sement mengde [kg]': 'Total grout take',
                        'Slutt trykk [bar]': 'Stop pressure',
                        'Forrige sementmengde': 'Prev. grout take',
                        'Forrige injeksjonstid': 'Prev. grouting time',
                        'Sementtype': 'Cement_type'})

df.to_excel(r'01_processed_data\outliers_removed.xlsx')

# plot a pairplot and corr heatmap of al inputs+output
if PAIRPLOT is True:
    pltr.seaborn_pairplot(df=df,
                          savepath=r'03_graphics/pairplot_reg.jpg',
                          to_plot=TO_PAIRPLOT+OUTPUT)
else:
    print('No pairplot created')

if HEATMAP is True:
    pltr.seaborn_corrplot(df=df,
                          savepath=r'03_graphics/feature_heatmap1.jpg')
else:
    print('No heatmap created')

##############################################################################
# objective definition for OPTUNA study


def objective(trial):
    # have optuna suggest a input feature combination
    input_index = int(trial.suggest_categorical('input feature combination',
                                                np.arange(0, len(input_)).astype(str)))

    # get input and output numpy arrays
    X = df[list(input_[input_index])].values
    y = df[OUTPUT].values

    # evtl. scaling of data
    scaler = trial.suggest_categorical('scaler',
                                       ['no scaler', 'MinMax', 'Standard'])
    if scaler == 'no scaler':
        pass
    elif scaler == 'MinMax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif scaler == 'Standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    y = np.squeeze(y)

    if OPTIMIZATION is False:
        # do the train test split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT,
                                                            random_state=42)
    
        # reduce unnecesarry dimensions (reguired for the RandomForest)
        y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)

    ###########################################################################
    # ML training and evaluation
    # definition of models

    # Choose which model to use
    if MODEL == 'RandomForest':
        regressor = RandomForestRegressor(n_estimators=trial.suggest_int('n_estimators_RF', 10, 200, step=10),
                                          criterion=trial.suggest_categorical('criterion_RF', ['squared_error', 'absolute_error', 'poisson']),
                                          min_samples_leaf=trial.suggest_int('min_samples_leaf_RF', 1, 20, step=1),
                                          bootstrap=trial.suggest_categorical('bootstrep_RF', [True, False]),
                                          # oob_score=trial.suggest_categorical('oob_score_RF',[True, False]),
                                          n_jobs=-1, random_state=42)
    elif MODEL == 'DecisionTree':
        regressor = DecisionTreeRegressor(criterion=trial.suggest_categorical('criterion_DT', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                                          splitter=trial.suggest_categorical('splitter', ['best', 'random']),
                                          min_samples_leaf=trial.suggest_int('min_samples_leaf_DT', 1, 20, step=1),
                                          random_state=42)
    elif MODEL == 'ExtraTrees':
        regressor = ExtraTreesRegressor(n_estimators=trial.suggest_int('n_estimators_ET', 10, 200, step=10),
                                        criterion=trial.suggest_categorical('criterion_ET', ['squared_error', 'absolute_error']),
                                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20, step=1),
                                        bootstrap=trial.suggest_categorical('bootstrep_ET', [True, False]),
                                        # oob_score=trial.suggest_categorical('oob_score_ET',[True, False]),
                                        n_jobs=-1, random_state=42)
        
    # TODO: Do the cross validation
    
    scores = cross_val_score(regressor, X, y, cv=4, scoring='r2')
    score = np.mean(scores)

    if OPTIMIZATION is False:

        # Fit regressort to training set
        regressor.fit(X_train, y_train)
    
        # Prediction of test data
        y_pred = regressor.predict(X_test)
    
        # Evaluation of model accuracy
        R2_reg = r2_score(y_test, y_pred)  # R2 score
        # MAXE_reg = max_error(y_test, y_pred) # Max error
        # MAE_regr = mean_absolute_error(y_test, y_pred)  # Mean absolute error

    if OPTIMIZATION is True:
        return score
    elif OPTIMIZATION is False:
        return score, R2_reg, regressor, y_test, y_pred, X_train


###############################################################################
# run optimization study or main run

if OPTIMIZATION is True:  # study
    try:  # evtl. load already existing study if one exists
        study = joblib.load(fr'02_results\{STUDY_NAME}.pkl')
        print('prev. study loaded')
    except FileNotFoundError:  # or create a new study
        study = optuna.create_study(direction='maximize')
        print('new study created')
    # the OPTUNA study is then run in a loop so that intermediate results are
    # saved and can be checked every 2 trials
    for _ in range(20):  # save every xx. study
        study.optimize(objective, n_trials=100)
        joblib.dump(study, fr"02_results\{STUDY_NAME}.pkl")

        df_study = study.trials_dataframe()
        df_study.to_csv(fr'02_results\{STUDY_NAME}.csv')
    # print results of study
    trial = study.best_trial
    print(f'\nhighest reward: {trial.value}')
    print(f'Best hyperparameters: {trial.params}')
    print('Best input feature combination:')
    print(input_[int(trial.params['input feature combination'])])
else:  # main run
    print('New main run created; no study results will be saved')
    study = joblib.load(fr'02_results\{STUDY_NAME}.pkl')
    trial = study.best_trial
    print(f'\nhighest reward: {trial.value}')
    print(f'Best hyperparameters: {trial.params}')
    print('Best input feature combination:')
    print(input_[int(trial.params['input feature combination'])])

    score, R2_reg, regressor, y_test, y_pred, X_train = objective(trial)

    print(f'R2 score: {R2_reg}, cross val R2: {score}')

# Plot for evaluation - prediction vs ground truth
# Remember to set figtext_y = 4000 if grout take and 5 if time consuption
pltr.result_scatter(y_test, y_pred=y_pred,
                    R2_score=round(R2_reg, 2),
                    cross_val=round(score, 2),
                    output_name=OUTPUT,
                    algorythm_name=MODEL,
                    figtext_y=5,
                    savepath=fr'03_graphics/{STUDY_NAME}_pred_vs_truth.jpg')

###############################################################################
# do optimization of grouting parameters

# Get input features used for the best trial
INPUT = input_[int(trial.params['input feature combination'])]

# Make a new DataFrame with columns = best trial input
df_optimization = pd.DataFrame(columns=INPUT)

# Get 100 values between optimization target min and max value
df_optimization[OPTIMIZATION_TARGET[0]] = np.linspace(df[OPTIMIZATION_TARGET[0]].min(),
                                                      df[OPTIMIZATION_TARGET[0]].max(),
                                                      num=100)

for feature in df[INPUT].drop(OPTIMIZATION_TARGET, axis=1).columns:
    df_optimization[feature] = np.full(len(df_optimization),
                                        df[feature].median())

X_optimize = df_optimization[INPUT].values
y_optimization = regressor.predict(X_optimize)

# Plot to check correlation between output and optimization target
pltr.optimization_plotter(df=df_optimization,
                          optimize_target=OPTIMIZATION_TARGET[0],
                          y_optimize=y_optimization, savepath=fr'03_graphics/{STUDY_NAME}_correlation.jpg')
##############################################################################
# Feature importances

# # Check importance of features by using plots
if MODEL == 'RandomForest':
    pltr.feature_importances(regressor=regressor, algorythm_name=MODEL,
                              output_name=OUTPUT, feature=INPUT,
                              savepath=fr'03_graphics/{STUDY_NAME}_feature_importance.jpg')

elif MODEL == 'ExtraTrees':
    pltr.feature_importances(regressor=regressor, algorythm_name=MODEL,
                              output_name=OUTPUT, feature=INPUT,
                              savepath=fr'03_graphics/{STUDY_NAME}_feature_importance.jpg')

elif MODEL == 'DecisionTree':
    pltr.feature_importances(regressor=regressor, algorythm_name=MODEL,
                              output_name=OUTPUT, feature=INPUT,
                              savepath=fr'03_graphics/{STUDY_NAME}_feature_importance.jpg')

