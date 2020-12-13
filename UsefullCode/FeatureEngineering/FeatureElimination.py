########################### Features elimination
from scipy.stats import ks_2samp
features_check = []
columns_to_check = set(list(train_f)).difference(base_columns+rm_cols)
for i in columns_to_check:
    features_check.append(ks_2samp(test_f[i], train_f[i])[1])

features_check = pd.Series(features_check, index=columns_to_check).sort_values()
features_discard = list(features_check[features_check==0].index)
print(features_discard)

# We will reset this list for now (use local test drop),
# Good droping will be in other kernels
# with better checking
features_discard = []

# Final features list
features_columns = [col for col in list(train_f) if col not in rm_cols + features_discard]




# ks features elimination

from scipy.stats import ks_2samp
list_p_value =[]

for i in tqdm(df_train_columns):
    list_p_value.append(ks_2samp(df_test[i] , df_train[i])[1])

Se = pd.Series(list_p_value, index = df_train_columns).sort_values()