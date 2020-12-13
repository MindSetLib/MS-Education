import numpy as np
import pandas as pd
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count


# df = pd.DataFrame(np.random.randint(3, 10, size=[50000, 2000]))

def func(col):
    print(col)
    for agg_type in ['mean', 'std', 'median', 'var']:
        try:

            new_col_name = col + '_TransactionAmt_' + agg_type
            temp_df = pd.concat([train_tr[[col, 'TransactionAmt']], train_tr[[col, 'TransactionAmt']]])
            # temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
            temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                columns={agg_type: new_col_name})

            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()

            # print(temp_df)

            # train_tr[new_col_name] = train_tr[col].map(temp_df)
            # test_df[new_col_name]  = test_df[col].map(temp_df)
        except:
            print('err', col)
    return train_tr[col].map(temp_df)


# categorical_columns = [c for c in train_tr.columns if train_tr[c].dtype.name == 'object']

i_cols = ['card1', 'card2', 'card3', 'card5', 'ProductCD', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
# _cols=list(train_tr.columns)

pool = Pool(cpu_count())  # Create a multiprocessing Pool
res = pool.map(func, i_cols)

# close down the pool and join
pool.close()
pool.join()
pool.clear()


#----------------------------------------------------

excluded_columns=['TransactionID', 'isFraud', 'TransactionDT']
initial_columns=list(train_tr.columns)
feature_columns=[x for x in initial_columns if x not in excluded_columns]


# choose a random element from a list
from random import seed
from random import choice
# seed random number generator
seed(1)
# prepare a sequence
sequence = feature_columns
#print(sequence)
# make choices from the sequence
local_cluster_cols=[]
general_cluster_cols=[]
general_df_cols=[]
for i in range (0,10):
  print('clustering', i)
  for _ in range(5):
    selection = choice(sequence)
    local_cluster_cols.append(selection)
    #print(selection)
  general_cluster_cols.append(local_cluster_cols)
  general_df_cols.append(train_tr[local_cluster_cols])
  local_cluster_cols=[]


  def func(col):
      num_cluster = 15
      # col.fillna(-1,inplace=True)
      clusterer = hdbscan.HDBSCAN(min_cluster_size=num_cluster, core_dist_n_jobs=4).fit(col)
      threshold = pd.Series(clusterer.outlier_scores_).quantile(0.90)
      outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
      return clusterer.labels_


  i_cols = general_df_cols

  pool = Pool(cpu_count())  # Create a multiprocessing Pool

  res = pool.map(func, i_cols)

  # close down the pool and join
  pool.close()
  pool.join()
  pool.clear()