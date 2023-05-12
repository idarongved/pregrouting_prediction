# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:43:20 2022

@author: IRo
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class plotter:

    def __init__(self, INPUT, OUTPUT):
        self.INPUT = INPUT
        self.OUTPUT = OUTPUT

    def custom_histograms(self, df, savepath):
        '''custom function to historgrams of datapoint for each feature'''
        fig = plt.figure(figsize=(len(self.INPUT + self.OUTPUT)*4, 4))

    # enumerate gives a counter. Rembember that matplotlib counts from 1, not 0
        for i, feature in enumerate(self.INPUT + self.OUTPUT):
            ax = fig.add_subplot(1, len(self.INPUT + self.OUTPUT), i+1)
            ax.hist(df[feature], bins=30, color='grey', edgecolor='black')
            # alpha gives the transparancy of the grid
            ax.grid(alpha=0.5)
            ax.set_xlabel(feature)
            if i == 0:
                ax.set_ylabel('n_datapoints')

        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()

    def result_scatter(self, y_test, y_pred, R2_score, cross_val,
                       algorythm_name, output_name, figtext_y, savepath):
        '''function to plot prediction (result) vs ground truth '''
        max_y_test = max(y_test)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(y_test, y_pred, label=algorythm_name)
        ax.plot((0, max_y_test), (0, max_y_test), color='black', alpha=0.7)

        min_limit = y_test.min() - y_test.max() * 0.1
        max_limit = y_test.max() + y_test.max() * 0.1
        ax.set_xlim(left=min_limit, right=max_limit)
        ax.set_ylim(bottom=min_limit, top=max_limit)

        ax.grid(alpha=0.5)
        ax.set_xlabel('Ground truth')
        ax.set_ylabel('Prediction')
        ax.legend()
        plt.tight_layout()
        plt.title(fr'{algorythm_name} predicting {output_name}', fontsize=10)
        plt.text(figtext_y, 0.25, fr'R2 score: {R2_score}, Cross val score: {cross_val}')
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    def optimization_plotter(self, df, optimize_target, y_optimize, savepath):
        '''function to plot model output vs optimization target'''
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(df[optimize_target], y_optimize)

        ax.set_xlabel(optimize_target)
        ax.set_ylabel(self.OUTPUT[0])
        ax.grid(alpha=0.5, color='grey')
        plt.tight_layout()
        plt.savefig(savepath, dpi=300, bbox_inches='tight')


    def feature_importances(self, regressor, algorythm_name, output_name,
                            feature, savepath):
        '''fuction to create barh plot of feature importances'''
        fig, ax = plt.subplots(figsize=(8, 4))
        importances = regressor.feature_importances_
        ax.bar(x=np.arange(len(importances)), height=importances,
               tick_label=feature)
        plt.xticks(rotation=45, ha='right')
        ax.set_title(f'Feature importace for: {algorythm_name} predicting {output_name}')
        ax.set_axisbelow(True)
        ax.grid(alpha=0.5)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        plt.tight_layout()
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        
    def seaborn_pairplot(self, df, savepath, to_plot):
        '''fuction that make pair plot of input and output features'''
        df.drop(labels=['Type injeksjon', 'Kontrollresultat', 'PegEnd',
                        'Bormeter [m]', 'Tunnel', 'PegStart', 'User', 'Date'],
                        axis=1, inplace=True)
        sns.pairplot(df, vars=to_plot, hue='Sementtype', kind='reg',
                     diag_kind='hist')
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        
    def seaborn_corrplot(self, df, savepath):
        '''function that make corr heatmap of input and output features '''
        df.drop(labels=['Type injeksjon', 'Kontrollresultat', 'PegEnd',
                        'Bormeter [m]', 'Tunnel', 'PegStart', 'User', 'Date',
                        'key_0', 'Jn', 'Jr', 'RQD', 'Jw', 'SRF', 'Ja', 'Stikning [m]',
                        'JnMult', 'Pel', 'Skjermlengde [m]', 'Sementtype'],
                        axis=1, inplace=True)
        plt.figure(figsize=(16, 6))
        mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
        heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1,
                              cmap="bwr", annot=True)
        heatmap.set_title('Triangle Correlation Heatmap',
                          fontdict={'fontsize':18}, pad=16)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')


class utilities(plotter):

    def __init__(self):
        pass

    # fuction scale the data from 0 to 1, and find max and min values
    def min_max_scaling(self, data, min_val=None, max_val=None):
        '''custom function for scaling data between 0 and 1'''
        if min_val is None:
            min_val = data.min(axis=0)
        data = data - min_val
        if max_val is None:
            max_val = data.max(axis=0)
        data = data / max_val
        return data, min_val, max_val

    def rescaling(self, data, min_val, max_val):
        data = data * max_val
        data = data + min_val
        return data

    def merge_inj(self, files, folder):
        '''function to merge files with injection data'''
        df_s = []

        for file in files:
            # Filter out water-file
            if 'Sikr' in file:
                # function that reads text files
                df_temp = pd.read_csv(fr'{folder}\{file}', skiprows=1, sep=';',
                                      decimal=',')
                # drop the columns that we dont need
                df_temp.drop(['Volum [l]', 'Boretid [time]', 'Kommentar',
                              'Unnamed: 18'], axis=1, inplace=True)
                # drops the rows with missing data
                df_temp.dropna(inplace=True)
                # only use type injeksjon = Forinjeksjon
                df_temp = df_temp[df_temp['Type injeksjon'] == 'Forinjeksjon']
                # move the df_temp to the empty list df_s
                df_s.append(df_temp)
        # merge (concatination) of all df_temp
        df = pd.concat(df_s)

        # arange the index in df
        df.index = np.arange(len(df))

        # making an excel-file
        df.to_excel(r'01_processed_data\Merged_files.xlsx')
        return df

    def remove_outliers(self, data, dictionary, before_after_plot=False):
        '''fuction to remove outliers by using dictionary and loops'''
        if before_after_plot is True:
            self.custom_histograms(data, savepath=r'03_graphics\histogramsbefore.pdf')

        # TODO (optional) make function more generalizable
        for key, val in dictionary.items():
            if val > 0:
                data = data[data[key] < val]
            if val < 0:
                key = 'WaterFlowNormMean'
                data = data[data[key] > val]

        if before_after_plot is True:
            self.custom_histograms(data, savepath=r'03_graphics\histogramsafter_1.pdf')

        return data

    def all_combinations(self, LIST, min_features=2):
        '''function that generates all possible combinations of a list with a
        minimum number of elements'''
        subsets = []
        for L in range(0, len(LIST)+1):
            for subset in itertools.combinations(LIST, L):
                if len(subset) >= min_features:
                    subsets.append(subset)
        return subsets
