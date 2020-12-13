#select features by condition

#variant 1
import pandas as pd

data = {'spike-2': [1,2,3], 'hey spke': [4,5,6], 'spiked-in': [7,8,9], 'no': [10,11,12]}
df = pd.DataFrame(data)

spike_cols = [col for col in df.columns if 'spike' in col]
print(list(df.columns))
print(spike_cols)

#variant 2
dataframe.filter(regex='^lifeExp',axis=1).head()




# example of target encoding
for col in ['ProductCD','M4']:
    temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
                                                        columns={'mean': col+'_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col+'_target_mean'].to_dict()

    train_df[col+'_target_mean'] = train_df[col].map(temp_dict)
    test_df[col+'_target_mean']  = test_df[col].map(temp_dict)


#get mean and stdev in encoding
i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']

for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col + '_TransactionAmt_' + agg_type
        temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col, 'TransactionAmt']]])
        # temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
            columns={agg_type: new_col_name})

        temp_df.index = list(temp_df[col])
        temp_df = temp_df[new_col_name].to_dict()

        train_df[new_col_name] = train_df[col].map(temp_df)
        test_df[new_col_name] = test_df[col].map(temp_df)