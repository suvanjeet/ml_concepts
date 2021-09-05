
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
from sklearn import preprocessing

#%%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

cd = pd.read_csv(r'C:\Users\suvanjeet\Desktop\customerfiles\New folder\FILE_CUST_DETAILS.csv',
                 sep=",", header= 0)
rem_td=[]
for a in range(0, len(cd)):
    if cd.iloc[a]['GDPR']=='N':
        rem_td.append(cd.iloc[a]['Cust_key'])

cd = cd[cd.GDPR == 'Y']
cd=cd.reset_index()
cd

#%%
td = pd.read_csv(r'C:\Users\suvanjeet\Desktop\customerfiles\New folder\Sales.csv',
                 sep=",", encoding = "ISO-8859-1")
td = td[td.retail_type == 'SALES']

for b in rem_td:
    td = td[td.cust_key != b]
print(td['cust_key'].unique())            
td=td.reset_index()
td.head()
#%%
min_class=td[['sku_item_key','class_name','product_name']]
min_class= min_class.drop_duplicates('product_name')
min_class=min_class.reset_index()
min_class=min_class.drop(['index'],axis=1)
min_class

#%%
prodname=td['product_name'].unique().tolist()
prodname=pd.DataFrame({'prod_name':prodname})

print(prodname)

prodname['offer'] = "2% off in "+ prodname['prod_name']
prodname['prc_red']= "2%"
prodname['valid_date']= '31-01-2019'
prodname['off_ref_id'] = np.random.randint(100000,500000, size=len(prodname))
print(prodname)

#%%
def chancetobuy(td,fday):
    tdd=td[['day_key','cust_key']]

    tdd=tdd.drop_duplicates()
    tdd['trans_amt']=0

    tdd=tdd.reset_index()
    tdd['index'] = range(1, len(tdd) + 1)

    for i in range(0, len(tdd)):
        for j in range(0,len(td)):
        
            if (tdd.iloc[i]['day_key'] == td.iloc[j]['day_key'] and tdd.iloc[i]['cust_key'] == td.iloc[j]['cust_key']):
                k=td.iloc[j]['net_amt']
                tdd.at[(tdd.iloc[i]['index']-1),'trans_amt'] = tdd.iloc[i]['trans_amt']+ k

    tdd1=tdd.drop(['index'],axis=1)
    tdd1[['day_key']] = tdd1[['day_key']].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[0:4],s[4:6],s[6:]))

    summary = summary_data_from_transaction_data(tdd1,
                                             customer_id_col = 'cust_key',
                                             datetime_col = 'day_key',
                                             monetary_value_col='trans_amt',
                                             observation_period_end='2018-12-30')

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
    top= segmented_rfm.sort_values('RFMScore').head(18)
    print(top)
    top.reset_index(inplace=True)
    #top['index'] = range(1, len(top) + 1)
    topcustid=top['cust_key'].tolist()
    #ID = top.loc[top['Cust_id']].tolist()
    print(topcustid)
    return topcustid

fday=30
ctb=chancetobuy(td,fday)

#%%
def DataClean(cd,td):
    
    

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    per_miss = cd.isnull().sum()/len(cd)

    missing_features = per_miss[per_miss > 0.50].index
    cd = cd.drop(missing_features, axis=1)
    print(cd)
    
    per_miss = td.isnull().sum()/len(td)
    missing_features = per_miss[per_miss > 0.50].index
    td = td.drop(missing_features, axis=1)
    print(td.head())

    cat_col=td['class_name'].unique()
    for i in cat_col:
        cd[i]=0
    cd1=cd
    
    cd1['index'] = range(1, len(cd1) + 1)
    for i in range(0, len(cd1)):
        for j in range(0,len(td)):
        
            if cd1.iloc[i]['Cust_key'] == td.iloc[j]['cust_key']:
                k=td.iloc[j]['class_name']
                cd1.at[(cd1.iloc[i]['index']-1), k] = cd1.iloc[i][k]+ td.iloc[j]['quantity']
    #print(cd1)
    cd2=cd1.drop(['Name','Cust_key','index','card_nbr','user_id','addr1','addr2',
                  'state','postcode','GDPR','DOB'],axis=1)
    col_obj=['point_type','card_type','city','Gender','Marital_Status']
    le = {}
    for a in col_obj:
        le[a]= preprocessing.LabelEncoder()
    for a in col_obj:
        cd2[a] = le[a].fit_transform(cd2[a])

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
    
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(cd2)
    cd1['clusters']= kmeans.labels_
    print(cd1[['Cust_key','clusters']])
    td1=td
    cd3=cd1
    
    
    final= pd.merge(td1,cd3[['Cust_key','clusters']], left_on='cust_key',right_on='Cust_key',how='left')
    #print(final.head(15))
    print(final[['cust_key','clusters']])
    final=final.drop(['Cust_key'],axis=1)
    print(final.head(15))
    return final    

cleandata=DataClean(cd,td)     
#cusid=[1,2,3,4]    

#%%

lastprod=td[['day_key','cust_key','product_name']]
lastprod=lastprod.sort_values('day_key')
lastprod=lastprod.reset_index()
lastprod=lastprod.drop(['index'],axis=1)
print(lastprod)
recent_custdate=lastprod[['day_key','cust_key']]
recent_custdate=recent_custdate.drop_duplicates()
print(recent_custdate)
recent_custdate=recent_custdate.sort_values('day_key', ascending=False)
print(recent_custdate)
recent_custdate=recent_custdate.drop_duplicates('cust_key')

recent_custdate=recent_custdate.reset_index()
recent_custdate=recent_custdate.drop(['index'],axis=1)
print(recent_custdate)

#%%   
def cluster(custid,ctb,recent_custdate,lastprod,cleandata,cd):

    final= cleandata
    print(final.columns)
    def prodfetch(custid):
        for r in range(0, len(recent_custdate)):
            if recent_custdate.iloc[r]['cust_key'] == custid:
                for s in range(0, len(lastprod)):
                    if recent_custdate.iloc[r]['cust_key']== lastprod.iloc[s]['cust_key'] and recent_custdate.iloc[r]['day_key'] == lastprod.iloc[s]['day_key']:
                        prod_list.append(lastprod.iloc[s]['product_name'])
                    
    
    print(ctb)
    print(final.head())    
    cust_cluster=final[['cust_key','clusters']]
    cust_cluster=cust_cluster.drop_duplicates()
    print(cust_cluster)
    df = pd.DataFrame()
    for i in custid:
        for j in range(0, len(cust_cluster)):
            if i== cust_cluster.iloc[j]['cust_key']:
                k= cust_cluster.iloc[j]['clusters']
                subsetDataFrame0 = final[final['clusters'] == k]
            
                sdf0=subsetDataFrame0[['cust_key','product_name']]
                #sdf0=subsetDataFrame0.drop(['day_key', 'transaction_id','customer_name','Category',
                #           'quantity','Total_Price','Status','clusters'], axis=1)
                print(sdf0.head())
                #Get list of unique items
                itemList=list(set(sdf0["product_name"].tolist()))

                #Get count of users
                userCount=len(set(sdf0["product_name"].tolist()))

                #Create an empty data frame to store item affinity scores for items.
                itemAffinity= pd.DataFrame(columns=('item1', 'item2', 'score'))
                rowCount=0
                print(itemAffinity)
                #For each item in the list, compare with other items.
                for ind1 in range(len(itemList)):
    
                    #Get list of users who bought this item 1.
                    item1Users = sdf0[sdf0.product_name==itemList[ind1]]["cust_key"].tolist()
                    #print("Item 1 ", item1Users)
    
                    #Get item 2 - items that are not item 1 or those that are not analyzed already.
                    for ind2 in range(ind1, len(itemList)):
                        if ( ind1 == ind2):
                            continue
       
                        #Get list of users who bought item 2
                        item2Users=sdf0[sdf0.product_name==itemList[ind2]]["cust_key"].tolist()
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
                
                prod_list=[]
                prodfetch(i)
                
                prod_list
                if i in ctb:
                    for u in prod_list:
                        searchItem=u
                        recoList=itemAffinity[itemAffinity.item1==searchItem]\
                        [["item2","score"]]\
                        .sort_values("score", ascending=[0])
                                
                        print("\nRecommendations for user id ", i)                        
                    
                        print("Recommendations for item ",u ,"\n", recoList)
                        prolist=[] 

                        for l in range(0, len(recoList)):
                            if recoList.iloc[l]['score']>=0.01:
                                prolist.append(recoList.iloc[l]['item2'])
                                
                        if not prolist:
                            print("no offer available")
                        
                        for m in range(0, len(prodname)):
                            for n in range(len(prolist)):
                                if prodname.iloc[m]['prod_name'] == prolist[n]:
                                    c_list = []
                                    a_list = []
                                    b_list = []
                                    off_per = []
                                    val_dt = []
                                    off_ref = []
                                    
                                    print ('offer available  ',  prodname.iloc[m]['offer'])
                                    
                                    pn=prodname.iloc[m]['prod_name']
                                    po=prodname.iloc[m]['offer']
                                    off=prodname.iloc[m]['prc_red']
                                    vd= prodname.iloc[m]['valid_date']
                                    off_id= prodname.iloc[m]['off_ref_id']
                                    
                                    off_ref.append(off_id)
                                    off_per.append(off)
                                    val_dt.append(vd)
                                    c_list.append(i)
                                    a_list.append(pn)
                                    b_list.append(po)
                                    if len(df) == 0:
                                        df=pd.DataFrame({'Customer_Id': c_list,'Price_Reduction':off_per,'Validity_Date':val_dt,'Product_Description': a_list,'Reward_Description': b_list,'Price_Reduction_Id':off_ref})

                                    else:
                                        df2=pd.DataFrame({'Customer_Id': c_list,'Price_Reduction':off_per,'Validity_Date':val_dt,'Product_Description': a_list,'Reward_Description': b_list,'Price_Reduction_Id':off_ref})
                                        df = df.append(df2,ignore_index=True)                                    
                else:
                    print('\nuser id ', i ,'is not a previlledged customer')

    df= pd.merge(df,cd[['Cust_key','Name','user_id']], left_on='Customer_Id',right_on='Cust_key',how='left')
    
    
    df['Latest GDPR flag']='Y'
    df['Store Specific Reduction Flag']='Y'
    df= pd.merge(df,min_class[['product_name','sku_item_key','class_name']], left_on='Product_Description',right_on='product_name',how='left')
    df.rename({'Name':'Customer_Name', 'user_id':'Customer_Loyalty_Id','sku_item_key':'MIN','class_name':'Group Name'}, axis=1, inplace=True)
    df=df.drop(['Cust_key','product_name'],axis=1)
    
    df=df[['Customer_Id','Customer_Name', 'Customer_Loyalty_Id','Latest GDPR flag','MIN', 'Price_Reduction', 'Validity_Date'
           ,'Store Specific Reduction Flag','Product_Description', 'Reward_Description' ,'Group Name','Price_Reduction_Id']]
    print(df)
    df.to_csv('offers_customers.csv')
#custid=recent_custdate['cust_key'].tolist()
custid=[1,3] 
cluster(custid,ctb,recent_custdate,lastprod,cleandata,cd)



#%%



