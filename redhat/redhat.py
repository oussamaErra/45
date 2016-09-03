import pandas as pd 
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score , roc_curve , auc , log_loss
from datetime import datetime
import time
import gc


gc.enable()
test = pd.read_csv('C:/Users/oussama/Documents/red hat/act_test.csv',header=0)
train = pd.read_csv('C:/Users/oussama/Documents/red hat/act_train.csv',header=0)
people = pd.read_csv('C:/Users/oussama/Documents/red hat/people.csv',header=0)
for x in train.columns:
    if train[x].isnull().sum().item()>1000:
        train.drop(x,inplace=True,axis=1)
        test.drop(x,inplace=True,axis=1)
for x in [ col for col in people.columns if people[col].dtype ==np.dtype(bool)]:
     people[x] = people[x]*1
for k in range(1,10,1):
    people['char_{}'.format(k)]= pd.factorize(people['char_{}'.format(k)])[0]

    
train['day_of_week']=train.loc[:,'date'].apply(lambda x : datetime.strptime(str(x) ,'%Y-%m-%d').weekday())
train['month']=train.loc[:,'date'].apply(lambda x : datetime.strptime(str(x) ,'%Y-%m-%d').strftime('%B'))
train['year']=train.loc[:,'date'].apply(lambda x : datetime.strptime(str(x) ,'%Y-%m-%d').strftime('%Y'))
train.date = train.date.apply(lambda x : datetime.strptime(x , '%Y-%m-%d'))
people.date=people.date.apply(lambda x : datetime.strptime(x ,'%Y-%m-%d'))
train = pd.concat([train , pd.get_dummies(train.activity_category)] , axis=1)
train = pd.concat([train , pd.get_dummies(train.month)] , axis=1)
train.drop('month',axis=1,inplace=True)
train = pd.concat([train , pd.get_dummies(train.year)] , axis=1)
train.drop('year',axis=1,inplace=True)
train = pd.concat([train , pd.get_dummies(train.day_of_week)] , axis=1)
train.drop('day_of_week',axis=1,inplace=True)
train_data=pd.merge(train,people,on='people_id')
del train,test
#perfome same activities in the same time:
group = pd.DataFrame(train_data.groupby(['people_id','date_x' ,'activity_category']).size())
group.columns=['count_activity']
people_2=[]
people_3=[]
people_4=[]

for pep , df in group.groupby(level = 0):
    if 2 in df.count_activity.values:
        people_2.append(pep)
    if 3 in df.count_activity.values:
        people_3.append(pep)
    if 4 in df.count_activity.values:
        people_4.append(pep)
del group
t=set(people_2)
train_data['t_2_activities'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))  
t=set(people_3)
train_data['t_3_activities'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))
t=set(people_4)
train_data['t_4_activities'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))
#selecte the same acitivitie more than one time not finished 
group = pd.DataFrame(train_data.groupby(['people_id','activity_category']).size())
group.columns=['act_count']
same_activ_2 =[]
same_activ_4 =[]
same_activ_6 =[]
same_activ_8 =[]
same_activ_10 =[]
for pep,df in group.groupby(level=0):
    if any(df.act_count.values >9) :
        same_activ_10.append(pep)
    elif any(df.act_count.values >7) :
        same_activ_8.append(pep)
    elif any(df.act_count.values >5) :
        same_activ_6.append(pep)
    elif any(df.act_count.values >3) :
        same_activ_4.append(pep) 
    elif any(df.act_count.values >1) :
        same_activ_2.append(pep)  
    else  :
        pass 
del group
t=set(same_activ_2)
train_data['same_activity_2'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))               
t=set(same_activ_4)
train_data['same_activity_4'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))
t=set(same_activ_6)
train_data['same_activity_6'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))
t=set(same_activ_8)
train_data['same_activity_8'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))
t=set(same_activ_10)
train_data['same_activity_10'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))
#if yes selecting them in the same time not finished 
activities_2=[]
activities_4=[]
activities_6=[]
activities_8=[]
activities_10=[]
tet = pd.DataFrame(train_data.groupby(['people_id','date_x'])['activity_category'].agg({'counts_the_activities':np.size}))
for pep , df in tet.groupby(level=0):
    if 2 & 3 in df.counts_the_activities.values:
        activities_2.append(pep)
    if 4 & 5 in df.counts_the_activities.values:
        activities_4.append(pep)
    if 6 & 7 in df.counts_the_activities.values:
        activities_6.append(pep)
    if 8 & 9 in df.counts_the_activities.values:
        activities_8.append(pep)
    if any(df.counts_the_activities.values>9):
        activities_10.append(pep)
del tet
t=set(activities_2)
train_data['same_time_activ_2'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))                
t=set(activities_4)
train_data['same_time_activ_4'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))                 
t=set(activities_6)
train_data['same_time_activ_6'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))                 
t=set(activities_8)
train_data['same_time_activ_8'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))                 
t=set(activities_10)
train_data['same_time_activ_10'] = train_data.people_id.apply(lambda x : set([x]).intersection(t)==set([x]))                 
# numbers of selected activitie per person
train_data['occur']=train_data.people_id
train_data.occur=train_data.people_id.apply(dict(train_data.people_id.value_counts() ).get)
#moyen of interval between activities 
for pep , df in train_data.groupby('people_id')['date_x']:
     df=pd.DataFrame(df)
     df.sort(columns='date_x' , ascending=False, inplace=True)
     l=list(set(df.date_x.values))
     if len(l)>1:
         mean_time= (sum([l[i]-l[i+1] for i in range(0,len(l)-1,1)])/np.timedelta64(1,'D'))/(len(df.date_x.values)-1)
         people.loc[people.people_id==pep,'mean_time']=mean_time
     else:
         people.loc[people.people_id==pep,'mean_time']=0

train_data=pd.merge(train_data,people.loc[:,['people_id','mean_time']],on='people_id')

#percentage of groups that are in the test and not in the train 
test_train.loc[test_train.group_1.isin(groups)==False,'group_1'].shape[0]/test_train.shape[0]
#first activities selected
first_activitie= train_data.loc[:,['people_id','date_x','activity_category']].sort(columns=['people_id','date_x']).drop_duplicates(['people_id'] ,keep='first')
first_activitie.rename(columns = {'activity_category':'first activity'} , inplace = True)
first_activitie.drop('date_x',axis=1,inplace=True)
last_activity = train_data.loc[:,['people_id','date_x','activity_category']].sort(columns=['people_id','date_x']).drop_duplicates(['people_id'],keep='last')
last_activity.rename(columns = {'activity_category':'last_activity'} , inplace=True)
last_activity.drop('date_x',axis=1,inplace=True)
train_data = pd.merge(train_data,first_activitie,on='people_id')
train_data=pd.merge(train_data,last_activity,on='people_id')
del last_activity , first_activitie
gc.collect()

#time between date_x and date y 
people_group =train_data.groupby('people_id')
frame_x=pd.DataFrame(people_group['date_x'].agg({'min_date_x':np.min}))
frame_y=pd.DataFrame(people_group['date_y'].agg({'min_date_y':np.min}))
frame_x.reset_index(level='people_id',inplace=True)
frame_y.reset_index(level='people_id',inplace=True)
frame=pd.merge(frame_x,frame_y,on='people_id')
frame['time_diff']=((frame.min_date_x -frame.min_date_y)/np.timedelta64(1,'D')).astype(int)
train_data=pd.merge(train_data,frame.loc[:,['people_id','time_diff']],on='people_id')
del people_group , frame
for x in [ col for col in train_data.columns if train_data[col].dtype ==np.dtype(bool)]:
     train_data[x] = train_data[x]*1
#drop and start the train 
train.drop(['activity_category','date'] , axis=1,inplace=True)
rf=RandomForestClassifier(n_estimators= 100 , n_jobs=-1)
split = StratifiedKFold(y, n_folds=2)


for k,(train,test) in enumerate(split):
    X_train,X_test,y_train,y_test = X[train] ,X[test],y[train] , y[test]
    rf.fit(X_train,y_train)
    gc.collect()
    print('the train accuracy of the {} fold is : {} '.format(k,rf.score(X_train,y_train)))
    print('the test accuracy of the  {} fold is : {}'.format(k,rf.score(X_test,y_test)))
    print('\n\n the train  log loss of the {} fold is : {}'.format(k,log_loss(y_train ,rf.predict_proba(X_train))))
    print('the test loss of the {} fold is : {}'.format(k,log_loss(y_test,rf.predict_proba(X_test))))
    fpr_train , tpr_train ,_train=roc_curve(y_train , rf.predict_proba(X_train)[:,1])
    fpr_test,tpr_test,_test= roc_curve(y_test,rf.predict_proba(X_test)[:,1])
    print('\n\n the train AUC score of the {} fold  is : {} '.format(k,auc(fpr_train,tpr_train)))
    print('the test AUC score of the {} fols is : {}'.format(k,auc(fpr_test,tpr_test)))
    print('\n')
    gc.collect()
    gc.collect()
    gc.collect()


print('------end of first trial---------')