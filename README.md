# PropertyPricePredictionModel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
df = pd.read_csv('/kaggle/input/hack-club-ai-ml-recruitments-ps-1-freshers/train.csv')
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar = False)
df.info()
df['Lot Frontage']=df['Lot Frontage'].fillna(df['Lot Frontage'].mean())
df.drop(['Alley'],axis=1,inplace=True)
df['Bsmt Cond']=df['Bsmt Cond'].fillna(df['Bsmt Cond'].mode()[0])
df['Bsmt Qual']=df['Bsmt Qual'].fillna(df['Bsmt Qual'].mode()[0])
df['Fireplace Qu']=df['Fireplace Qu'].fillna(df['Fireplace Qu'].mode()[0])
df['Garage Type']=df['Garage Type'].fillna(df['Garage Type'].mode()[0])
df.drop(['Garage Yr Blt'],axis=1,inplace=True)
df['Garage Finish']=df['Garage Finish'].fillna(df['Garage Finish'].mode()[0])
df['Garage Qual']=df['Garage Qual'].fillna(df['Garage Qual'].mode()[0])
df['Garage Cond']=df['Garage Cond'].fillna(df['Garage Cond'].mode()[0])
df.drop(['Pool QC','Fence','Misc Feature'],axis=1,inplace=True)
df.drop(['Order'],axis=1,inplace=True)
df.isnull().sum()
df['Mas Vnr Type']=df['Mas Vnr Type'].fillna(df['Mas Vnr Type'].mode()[0])
df['Mas Vnr Area']=df['Mas Vnr Area'].fillna(df['Mas Vnr Area'].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
df['Bsmt Exposure']=df['Bsmt Exposure'].fillna(df['Bsmt Exposure'].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
df['BsmtFin Type 2']=df['BsmtFin Type 2'].fillna(df['BsmtFin Type 2'].mode()[0])
df.dropna(inplace=True)
df.shape
columns=['MS Zoning','Street','Lot Shape','Land Contour','Utilities','Lot Config','Land Slope','Neighborhood',
         'Condition 2','Bldg Type','Condition 1','House Style','Sale Type',
        'Sale Condition','Exter Cond',
         'Exter Qual','Foundation','Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1','BsmtFin Type 2',
        'Roof Style','Roof Matl','Exterior 1st','Exterior 2nd','Mas Vnr Type','Heating','Heating QC',
         'Central Air',
         'Electrical','Kitchen Qual','Functional',
         'Fireplace Qu','Garage Type','Garage Finish','Garage Qual','Garage Cond','Paved Drive']
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:

        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)

        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:

            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1


    df_final=pd.concat([final_df,df_final],axis=1)
    main_df=df.copy()
   test_df=pd.read_csv('/kaggle/input/hack-club-ai-ml-recruitments-ps-1-freshers/train.csv')
test_df['Lot Frontage']=test_df['Lot Frontage'].fillna(test_df['Lot Frontage'].mean())
test_df['MS Zoning']=test_df['MS Zoning'].fillna(test_df['MS Zoning'].mode()[0])
test_df.drop(['Alley'],axis=1,inplace=True)
test_df['Bsmt Cond']=test_df['Bsmt Cond'].fillna(test_df['Bsmt Cond'].mode()[0])
test_df['Bsmt Qual']=test_df['Bsmt Qual'].fillna(test_df['Bsmt Qual'].mode()[0])
test_df['Fireplace Qu']=test_df['Fireplace Qu'].fillna(test_df['Fireplace Qu'].mode()[0])
test_df['Garage Type']=test_df['Garage Type'].fillna(test_df['Garage Type'].mode()[0])
test_df.drop(['Garage Yr Blt'],axis=1,inplace=True)
test_df['Garage Finish']=test_df['Garage Finish'].fillna(test_df['Garage Finish'].mode()[0])
test_df['Garage Qual']=test_df['Garage Qual'].fillna(test_df['Garage Qual'].mode()[0])
test_df['Garage Cond']=test_df['Garage Cond'].fillna(test_df['Garage Cond'].mode()[0])

test_df.drop(['Pool QC','Fence','Misc Feature'],axis=1,inplace=True)

test_df['Mas Vnr Type']=test_df['Mas Vnr Type'].fillna(test_df['Mas Vnr Type'].mode()[0])
test_df['Mas Vnr Area']=test_df['Mas Vnr Area'].fillna(test_df['Mas Vnr Area'].mode()[0])
test_df['Bsmt Exposure']=test_df['Bsmt Exposure'].fillna(test_df['Bsmt Exposure'].mode()[0])
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test_df['BsmtFin Type 2']=test_df['BsmtFin Type 2'].fillna(test_df['BsmtFin Type 2'].mode()[0])
test_df.loc[:, test_df.isnull().any()].head()
test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['Exterior 1st']=test_df['Exterior 1st'].fillna(test_df['Exterior 1st'].mode()[0])
test_df['Exterior 2nd']=test_df['Exterior 2nd'].fillna(test_df['Exterior 2nd'].mode()[0])
test_df['BsmtFin Type 1']=test_df['BsmtFin Type 1'].fillna(test_df['BsmtFin Type 1'].mode()[0])
test_df['BsmtFin SF 1']=test_df['BsmtFin SF 1'].fillna(test_df['BsmtFin SF 1'].mean())
test_df['BsmtFin SF 2']=test_df['BsmtFin SF 2'].fillna(test_df['BsmtFin SF 2'].mean())
test_df['Bsmt Unf SF']=test_df['Bsmt Unf SF'].fillna(test_df['Bsmt Unf SF'].mean())
test_df['Total Bsmt SF']=test_df['Total Bsmt SF'].fillna(test_df['Total Bsmt SF'].mean())
test_df['Bsmt Full Bath']=test_df['Bsmt Full Bath'].fillna(test_df['Bsmt Full Bath'].mode()[0])
test_df['Bsmt Half Bath']=test_df['Bsmt Half Bath'].fillna(test_df['Bsmt Half Bath'].mode()[0])
test_df['Kitchen Qual']=test_df['Kitchen Qual'].fillna(test_df['Kitchen Qual'].mode()[0])
test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['Garage Cars']=test_df['Garage Cars'].fillna(test_df['Garage Cars'].mean())
test_df['Garage Area']=test_df['Garage Area'].fillna(test_df['Garage Area'].mean())
test_df['Sale Type']=test_df['Sale Type'].fillna(test_df['Sale Type'].mode()[0])
test_df.to_csv('formulatedtest.csv',index=False)
test_df = df.copy()
final_df=pd.concat([df,test_df],axis=0)
final_df['SalePrice']
final_df=category_onehot_multcols(columns)
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
final_df
df_Train=final_df.iloc[:2636,:]
df_Test=final_df.iloc[2636:,:]
df_Test.drop(['SalePrice'],axis=1,inplace=True)
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
df_Test.shape
import xgboost
classifier = xgboost.XGBRegressor()
classifier.fit(X_train,y_train )
import pickle
filename = 'finalized_model.pk1'
pickle.dump(classifier,open(filename,'wb'))

y_pred = classifier.predict(df_Test)
pred= pd.DataFrame(y_pred)
sub_df=pd.read_csv('sample-submission.csv')
datasets=pd.concat([sub_df['Order'],pred],axis=1)
datasets.columns=['Order','SalePrice']
datasets.to_csv('sample-submission.csv',index=False)

    return df_final
