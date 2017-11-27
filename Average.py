import numpy as np 
import pandas as pd 

sub_1 = pd.read_csv('../Submissions/Newsub.csv')
sub_2 = pd.read_csv('../Submissions/MyLGBM3.csv')
sub_3 = pd.read_csv('../Submissions/MyLGBM0.csv')
sub_4 = pd.read_csv('../Submissions/MyLGBM_gini_.csv')
sub_5 = pd.read_csv('../Submissions/blend1.csv')
sub_6 = pd.read_csv('../Submissions/DartLGBM.csv')

sub = pd.DataFrame()
sub['id'] = sub_1['id']
sub['target'] = np.exp(np.mean([sub_1['target'].apply(lambda x: np.log(x)), sub_2['target'].apply(lambda x: np.log(x)), sub_3['target'].apply(lambda x: np.log(x)), sub_4['target'].apply(lambda x: np.log(x)), sub_5['target'].apply(lambda x: np.log(x)), sub_6['target'].apply(lambda x: np.log(x))], axis =0))

sub.to_csv('average.csv', index = False)