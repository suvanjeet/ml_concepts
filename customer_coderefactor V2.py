
#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import lifetimes
from lifetimes.plotting import *
from lifetimes.utils import *
from lifetimes.estimation import *
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter

#%%
cd = pd.read_csv(r'C:\Users\suvanjeet\Desktop\customerfiles\CUSTOMER_DETAILS.csv',
                 sep=",", header= 0)

td = pd.read_csv(r'C:\Users\suvanjeet\Desktop\customerfiles\transaction_details.csv',
                 sep=",", header= 0)

promodetails = pd.read_csv(r'C:\Users\suvanjeet\Desktop\customerfiles\promotion_details.csv',
                 sep=",", header= 0)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%

def chancetobuy(fday,td):
    tdd=td[['day_key','Cust_id']]

    tdd=tdd.drop_duplicates()
    tdd['trans_amt']=0

    tdd=tdd.reset_index()
    tdd['index'] = range(1, len(tdd) + 1)

    for i in range(0, len(tdd)):
        for j in range(0,len(td)):
        
            if (tdd.iloc[i]['day_key'] == td.iloc[j]['day_key'] and tdd.iloc[i]['Cust_id'] == td.iloc[j]['Cust_id']):
                k=td.iloc[j]['Total_Price']
                tdd.at[(tdd.iloc[i]['index']-1),'trans_amt'] = tdd.iloc[i]['trans_amt']+ k

    tdd1=tdd.drop(['index'],axis=1)
    tdd1[['day_key']] = tdd1[['day_key']].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[0:4],s[4:6],s[6:]))


    summary = summary_data_from_transaction_data(tdd1,
                                             customer_id_col = 'Cust_id',
                                             datetime_col = 'day_key',
                                             monetary_value_col='trans_amt',
                                             observation_period_end='2015-10-02')

    print(summary)

    bgf = BetaGeoFitter()
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])


    bgf.conditional_expected_number_of_purchases_up_to_time(fday,
                                                        summary['frequency'],
                                                        summary['recency'],
                                                        summary['T'])

    returning_customers_summary = summary[summary['frequency']>0]
    print(returning_customers_summary)
    print(len(returning_customers_summary))
    
    quantiles = returning_customers_summary.quantile(q=[0.25,0.5,0.75])
    quantiles = quantiles.to_dict()
    segmented_rfm = returning_customers_summary
    def RScore(x,p,d):
        if x <= d[p][0.25]:
            return 1
        elif x <= d[p][0.50]:
            return 2
        elif x <= d[p][0.75]: 
            return 3
        else:
            return 4
    
    def FMScore(x,p,d):
        if x <= d[p][0.25]:
            return 4
        elif x <= d[p][0.50]:
            return 3
        elif x <= d[p][0.75]: 
            return 2
        else:
            return 1
    
    
    segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
    segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
    segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))
    segmented_rfm.head()
    
    segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)
    print(segmented_rfm)
    
    #find top 10 best customers
    print(segmented_rfm.sort_values('RFMScore'))
    #http://databoosting.com/customer-segmentation-using-rfm/
    top= segmented_rfm.sort_values('RFMScore').head(4)
    print(top)
    top.reset_index(inplace=True)
    #top['index'] = range(1, len(top) + 1)
    topcustid=top['Cust_id'].tolist()
    #ID = top.loc[top['Cust_id']].tolist()
    print(topcustid)
    return topcustid
    

fday=30
ctb=chancetobuy(fday,td)


#%%
def DataClean(cd,td):
    
    

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(cd.head())
    per_miss = cd.isnull().sum()/len(cd)

    missing_features = per_miss[per_miss > 0.50].index
    cd = cd.drop(missing_features, axis=1)
    print(cd.head())
    
    
    print(td.head())
    per_miss = td.isnull().sum()/len(td)
    missing_features = per_miss[per_miss > 0.50].index
    td = td.drop(missing_features, axis=1)
    td = td[td.Status == 'sales']
    print(td.head())

    cat_col=td['Category'].unique()
    for i in cat_col:
        cd[i]=0
    cd1=cd
    cd1['index'] = range(1, len(cd1) + 1)
    for i in range(0, len(cd1)):
        for j in range(0,len(td)):
        
            if cd1.iloc[i]['CUSTOMER_ID'] == td.iloc[j]['Cust_id']:
                k=td.iloc[j]['Category']
                cd1.at[(cd1.iloc[i]['index']-1), k] = cd1.iloc[i][k]+ td.iloc[j]['quantity']
    print(cd1)
    cd2=cd1.drop(['CUSTOMER_NAME','CUSTOMER_ID','index'],axis=1)
    labelEncoder = LabelEncoder()
    labelEncoder.fit(cd2['SEX'])
    cd2['SEX'] = labelEncoder.transform(cd2['SEX'])
    
    sse = {}
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(cd2)
        cd2["clusters"] = kmeans.labels_
        #print(data["clusters"])
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()
    
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(cd2)
    cd1['clusters']= kmeans.labels_
    td1=td
    cd3=cd1
    
    
    final= pd.merge(td1,cd3[['CUSTOMER_ID','clusters']], left_on='Cust_id',right_on='CUSTOMER_ID',
                how='left')
    final=final.drop(['CUSTOMER_ID'],axis=1)
    print(final.head())

    #final['clusters'].unique()
    final.to_csv('finalsheet.csv')    

cleandata=DataClean(cd,td)     
#cusid=[1,2,3,4]    
#%%   
def cluster(custid,ctb):
    
    print(ctb)
    final = pd.read_csv(r'C:\Users\suvanjeet\finalsheet.csv',
                 sep=",", header= 0)
    #final = pd.read_csv(r'C:\Users\suvanje13799\finalsheet.csv',
                 #sep=",", header= 0)
    final=final.drop(final.columns[[0]], axis=1)
    print(final)
    cust_cluster=final[['Cust_id','clusters']]
    cust_cluster=cust_cluster.drop_duplicates()
    print(cust_cluster)
    for i in custid:
        for j in range(0, len(cust_cluster)):
            if i== cust_cluster.iloc[j]['Cust_id']:
                k= cust_cluster.iloc[j]['clusters']
                subsetDataFrame0 = final[final['clusters'] == k]
            
                sdf0=subsetDataFrame0.drop(['day_key', 'transaction_id','customer_name','Category',
                            'quantity','Total_Price','Status','clusters'], axis=1)
    
                #Get list of unique items
                itemList=list(set(sdf0["Product_name"].tolist()))

                #Get count of users
                userCount=len(set(sdf0["Product_name"].tolist()))

                #Create an empty data frame to store item affinity scores for items.
                itemAffinity= pd.DataFrame(columns=('item1', 'item2', 'score'))
                rowCount=0
                #For each item in the list, compare with other items.
                for ind1 in range(len(itemList)):
    
                    #Get list of users who bought this item 1.
                    item1Users = sdf0[sdf0.Product_name==itemList[ind1]]["Cust_id"].tolist()
                    #print("Item 1 ", item1Users)
    
                    #Get item 2 - items that are not item 1 or those that are not analyzed already.
                    for ind2 in range(ind1, len(itemList)):
                        if ( ind1 == ind2):
                            continue
       
                        #Get list of users who bought item 2
                        item2Users=sdf0[sdf0.Product_name==itemList[ind2]]["Cust_id"].tolist()
                        #print("Item 2",item2Users)
        
                        #Find score. Find the common list of users and divide it by the total users.
                        commonUsers= len(set(item1Users).intersection(set(item2Users)))
                        score=commonUsers / userCount

                        #Add a score for item 1, item 2
                        itemAffinity.loc[rowCount] = [itemList[ind1],itemList[ind2],score]
                        rowCount +=1
                        #Add a score for item2, item 1. The same score would apply irrespective of the sequence.
                        itemAffinity.loc[rowCount] = [itemList[ind2],itemList[ind1],score]
                        rowCount +=1


                #print(itemAffinity)
                searchItem='PEN'
                recoList=itemAffinity[itemAffinity.item1==searchItem]\
                [["item2","score"]]\
                .sort_values("score", ascending=[0])
                                
                print("\nRecommendations for user id ", i)
                #for p in ctb:
                if i in ctb:
                    print("Recommendations for item PEN\n", recoList)
                    prolist=[] 

                    for l in range(0, len(recoList)):
                        if recoList.iloc[l]['score']>=0.1:
                            prolist.append(recoList.iloc[l]['item2'])
                                
                    if not prolist:
                        print("no offer available")

                    for m in range(0, len(promodetails)):
                        for n in range(len(prolist)):
                            if promodetails.iloc[m]['product'] == prolist[n]:
                                print ('offer available  ',  promodetails.iloc[m]['offer'])
                else:
                    print('not a previlledged customer')
        
                #Check final result
                #print(itemAffinity)
            
   
custid=[1,2,5,7,9,11]   
cluster(custid,ctb)
  
#%%
lastprod=td[['day_key','Cust_id','Product_name']]
lastprod=lastprod.sort_values('day_key')
lastprod=lastprod.reset_index()
lastprod=lastprod.drop(['index'],axis=1)
print(lastprod)












