import numpy as np
import pandas as pd
import datetime as dt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.sparse as sparse

import keras as ks

import tensorrec
from tensorrec import TensorRec
from tensorrec.eval import *
from tensorrec.util import append_to_string_at_point

from tensorrec.representation_graphs import (AbstractKerasRepresentationGraph, 
#                                              ReLURepresentationGraph, 
                                             NormalizedLinearRepresentationGraph, 
#                                              LinearRepresentationGraph
                                            )

from tensorrec.loss_graphs import (WMRBLossGraph, 
#                                    BalancedWMRBLossGraph,
#                                    RMSELossGraph, 
#                                    RMSEDenseLossGraph, 
#                                    SeparationDenseLossGraph
                                  ) 
from tensorrec.eval import fit_and_eval, eval_random_ranks_on_dataset
from tensorrec.util import append_to_string_at_point

now = dt.date(2014, 3, 1)

class DeepRepresentationGraph(AbstractKerasRepresentationGraph):
    # 이 method는 사용자/아이템 특징을 사용자/아이템에 연결하는 정렬된 리스트의 keras 레이어를 반환.
    # representation. TensorRec이 훈련이 시작되면 해당 레이어를 이용하여 훈련 진행.
    def create_layers(self, n_features, n_components):
        return [
            ks.layers.Dense(n_components * 16, activation='relu'), # rectified linear unit(ReLU)
            ks.layers.Dense(n_components * 8, activation='relu'), # 다른 활성화 함수 시도 가능 
            ks.layers.Dense(n_components * 2, activation='relu'), # 대부분의 활성화 함수 변경은 이미지 인식과 같은 ML 경우에서 도움.
            ks.layers.Dense(n_components, activation='tanh'),
        ]

### recency score: 오래될 수록 점수가 높음
def RecencyScore(i, col, df):
    if i <= df[col][0.25]:
        return 4
    elif i <= df[col][0.50]:
        return 3
    elif i <= df[col][0.75]: 
        return 2
    else:
        return 1
    
# F, M, V: Recency와 달리, 사분위수가 높을 수록 점수가 높다
def FMVScore(i, col, df):
    if i <= df[col][0.25]:
        return 1
    elif i <= df[col][0.50]:
        return 2
    elif i <= df[col][0.75]: 
        return 3    
    else:
        return 4

def interaction_masking(interactions):
    '''
    This function will "mask" (a.k.a "hide") 20% of original interactions
    Masked items wil be considered not purchased

    '''
    mask_size = len(interactions.data)
    mask = np.random.choice(a=[False, True], size=mask_size, p=[.2, .8])
    not_mask = np.invert(mask)

    train_interactions = sparse.coo_matrix((interactions.data[mask],
                                        (interactions.row[mask],
                                         interactions.col[mask])),
                                       shape=interactions.shape)

    test_interactions = sparse.coo_matrix((interactions.data[not_mask],
                                       (interactions.row[not_mask],
                                        interactions.col[not_mask])),
                                      shape=interactions.shape)

    return train_interactions, test_interactions




def main():
    # 데이터 로드

    ## 마스터 데이터(상호 작용)
    masterdf = pd.read_csv('./data/Transactions.csv')
    masterdf.columns = ['Transaction ID', 'Customer ID', 'Transaction Date', 'Prod Subcat Code',
            'Prod Cat Code', 'Qty', 'Rate', 'Tax', 'Total Amt', 'Store Type'] # 데이터 정리 및 표준화를 위해 데이터 열 명칭 변경

    masterdf['Store Type Code'] = pd.factorize(masterdf['Store Type'])[0] # 상점 코드 타입을 숫자형으로 변경하여 새 열에 저장

    masterdf['Date'] =  pd.DatetimeIndex(masterdf['Transaction Date'], dayfirst=True).date # 거래 날짜를 pandas의 datetime index로 표준화 

    masterdf['Net Sales'] = masterdf['Qty'] * masterdf['Rate'] # quantity와 based price에서 총 순 매출액(Net sales) 계산 (도시마다의 세금이 다를 수 있어 세금 제외)

    masterdf['Material'] = masterdf['Prod Cat Code'].astype(str) + '-' + masterdf['Prod Subcat Code'].astype(str) + '-' + masterdf['Store Type'].astype(str) # category, subcategory, store type을 이용하여 고유한 material 표시기를 생성
    masterdf[['Prod Cat Code','Prod Subcat Code', 'Store Type', 'Material']].drop_duplicates(subset='Material')

    ## 소비자 데이터(소비자 특성)
    custdf = pd.read_csv('./data/Customer.csv')
    custdf.columns = ['Customer ID', 'DOB', 'Gender', 'City Code']

    ## 아이탬 특징 데이터
    skudf = pd.read_csv('./data/prod_cat_info.csv')
    skudf.columns = ['Prod Cat Code', 'Prod Cat', 'Prod Sub Cat Code', 'Prod Subcat']



    # 데이터 생성

    ## RECENCY (최신성)
    recency_df = masterdf.groupby('Customer ID').Date.max().reset_index()
    recency_df.columns = ['Customer ID','Last Purchase']
    recency_df['Recency'] = recency_df['Last Purchase'].apply(lambda x: (now - x).days)
    recency_df = recency_df[['Customer ID', 'Recency']]

    ## FREQUENCY (빈도)
    frequency_df = masterdf.groupby('Customer ID')['Date'].count().reset_index()
    frequency_df.columns = ['Customer ID','Frequency']

    ## MONETARY (금액)
    monetary_df = masterdf.groupby('Customer ID')['Net Sales'].sum().reset_index()
    monetary_df.columns = ['Customer ID','Monetary']

    ## VARIETY (종류)
    variety_df = masterdf.groupby('Customer ID')['Material'].nunique().reset_index()
    variety_df.columns = ['Customer ID','Variety']

    ## RFMV
    rfmv = recency_df.copy()
    rfmv = rfmv.merge(frequency_df, on='Customer ID')
    rfmv = rfmv.merge(monetary_df, on='Customer ID')
    rfmv = rfmv.merge(variety_df, on='Customer ID')

    rfmv_quantiles = rfmv.iloc[:, 1:].quantile(q = [0.25, 0.5, 0.75]).to_dict() # R, F, M, V의 25%, 50%, 75%의 사분위수를 dictonary 형식으로 저장

    rfmv2 = rfmv.copy()
    rfmv2['R_q'] = rfmv2['Recency'].apply(RecencyScore, args=('Recency', rfmv_quantiles ))
    rfmv2['F_q'] = rfmv2['Frequency'].apply(FMVScore, args=('Frequency', rfmv_quantiles ))
    rfmv2['M_q'] = rfmv2['Monetary'].apply(FMVScore, args=('Monetary', rfmv_quantiles ))
    rfmv2['V_q'] = rfmv2['Variety'].apply(FMVScore, args=('Variety', rfmv_quantiles ))

    rfmv2 = rfmv2[['Customer ID', 'R_q', 'F_q', 'M_q', 'V_q',]]

    ## 각 구성 요소의 총 점수 합계

    rfmv2['Total_Score'] = rfmv2['R_q'] + rfmv2['F_q'] + rfmv2['M_q'] + rfmv2['V_q']

    rfmv2 = rfmv2[['Customer ID', 'Total_Score']]

    # 중요(IMPORTANT) : 인덱스를 고객 번호로 설정
    rfmv2.index = rfmv2['Customer ID']
    rfmv2 = rfmv2.drop('Customer ID', 1)

    # 최적의 군집 수를 찾기 위해 elbow 방식 (차후 이 과정을 조정할 필요가 있음)
    wcss = []
    for i in range(2,10):
        kmeans = KMeans(n_clusters=i, 
                        init='k-means++')
        kmeans.fit(rfmv2)
        wcss.append(kmeans.inertia_)
        
    # 위 "elbow" 그래프의 최적의 수를 이용하여 KMean 군집 적용
    kmeans = KMeans(n_clusters=4, 
                    init='random', 
                    random_state=None)

    clusters = kmeans.fit_predict(rfmv2)

    ### 군집 결과를 원본 rfmv 데이터에 추가
    rfmv['Clusters'] = clusters


    # Recommendation Weight
    active_cust = rfmv[rfmv.Recency < 365] # 최근 1년(365일)을 기준으로 하여 실고객에게 추천

    cleaned_df = masterdf.merge(active_cust[['Customer ID','Clusters']], how='left', on='Customer ID') # 군집화된 고객 특징을 마스터 데이터에 결합
    cleaned_df = cleaned_df[cleaned_df['Clusters'].notnull()] # 군집을 기준으로 null 값이 존재하는 행 삭제
    cleaned_df = cleaned_df.merge(custdf[['Customer ID', 'City Code']], how='left', on='Customer ID') ## 소비자 데이터 추가
    cleaned_df = cleaned_df.merge(skudf[['Prod Cat', 'Prod Cat Code']], how='left', on='Prod Cat Code') # sku 특징(물품 카테고리) 를 마스터 데이터에 결합

    # 필수 열 가져오기
    final_cleaned_df = cleaned_df
    final_cleaned_df = final_cleaned_df[['Prod Cat','Material','Qty','Customer ID','Clusters',]]

    # 고유한 고객 목록 유지, 중복 제거
    cust_grouped = final_cleaned_df.groupby(['Customer ID',
                                            'Prod Cat',
                                            'Material',
                                            'Clusters']).sum().reset_index()

    ## Interaction Matrix 
    interactions = cust_grouped.groupby(['Customer ID', 'Material'])['Qty'].sum().unstack().fillna(0)

    minmaxscaler = preprocessing.MinMaxScaler()
    interactions_scaled = minmaxscaler.fit_transform(interactions)
    interactions_scaled = pd.DataFrame(interactions_scaled)

    interactions_scaled.index = interactions.index
    interactions_scaled.columns = interactions.columns

    ## User Features Matrix 
    cust_qty = cust_grouped.groupby(['Customer ID', 'Prod Cat'])['Qty'].sum().unstack().fillna(0)

    minmaxscaler = preprocessing.MinMaxScaler()
    cust_qty_scaled = minmaxscaler.fit_transform(cust_qty)
    cust_qty_scaled = pd.DataFrame(cust_qty_scaled)
    cust_qty_scaled.index = cust_qty.index
    cust_qty_scaled.columns = cust_qty.columns

    cust_clus = cust_grouped.groupby(['Customer ID', 'Clusters'])['Clusters'].nunique().unstack().fillna(0)

    customer_features = pd.merge(cust_qty_scaled, cust_clus, left_index=True, right_index=True, how='inner')
    customer_features = customer_features.rename(columns={0: 'Cluster 0', 
                                                        1: 'Cluster 1', 
                                                        2: 'Cluster 2', 
                                                        3: 'Cluster 3', 
                                                        4: 'Cluster 4'})

    ### Item Features Matrix
    item_category = pd.DataFrame(cust_grouped.groupby(['Material', 
                                                'Prod Cat'])['Qty'].sum().unstack().fillna(0).reset_index().set_index('Material'))

    minmaxscaler = preprocessing.MinMaxScaler()
    item_category_scaled = minmaxscaler.fit_transform(item_category)
    item_category_scaled = pd.DataFrame(item_category_scaled)
    item_category_scaled.index = item_category.index
    item_category_scaled.columns = item_category.columns


    interaction_f = sparse.coo_matrix(interactions_scaled)
    user_f  = sparse.coo_matrix(customer_features) 
    item_f  = sparse.coo_matrix(item_category_scaled) 

    mask_size = len(interaction_f.data)

    np.random.choice(a=[False, True], 
                    size=mask_size, 
                    p=[.2, .8])

    ## train, test data
    train_interactions, test_interactions = interaction_masking(interaction_f)

    user_features  = user_f
    item_features = item_f


    # train 

    ## 모델 파라미터
    epochs = 100 
    alpha = 0.01 
    n_components =  10

    verbose = True
    learning_rate = 0.01
    n_sampled_items = int(item_features.shape[0] * .1)
    biased = False
    
    k_val  = 100


    model = TensorRec(n_components = n_components,                 
                    user_repr_graph = DeepRepresentationGraph(),
                    item_repr_graph = NormalizedLinearRepresentationGraph(),
                    loss_graph = WMRBLossGraph(), 
                    biased=biased)

    model.fit(train_interactions, 
            user_features, 
            item_features, 
            epochs=epochs, 
            verbose=False, 
            alpha=alpha, 
            n_sampled_items=n_sampled_items,
            learning_rate=learning_rate)


    predicted_ranks = model.predict_rank(user_features=user_features,
                                        item_features=item_features)

    r_at_k_test = recall_at_k(predicted_ranks, test_interactions, k=80)
    r_at_k_train = recall_at_k(predicted_ranks, train_interactions, k=80)
    print("Recall at @k: Train: {:.2f} Test: {:.2f}".format(r_at_k_train.mean(), r_at_k_test.mean()))

    # produce the ranking into a readable table (dataframe it is)
    ranks_df = pd.DataFrame(predicted_ranks)
    ranks_df.columns = item_category_scaled.index
    ranks_df.index = customer_features.index
    ranks_df = ranks_df.T

    ranks_df.to_csv('./result/ranks_df.csv')

if __name__ == '__main__':
    main()