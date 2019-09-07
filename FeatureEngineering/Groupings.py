
# создание групп по бинам
def create_bin_groups(df,binnums=100,TARGET):
    temp_list=list(train_tr.columns)
    temp_list.remove(TARGET)
    df_same=df.copy()
    for col in temp_list:
        try:
            ser, bins = pd.qcut(df[col].values, binnums,duplicates ='drop', retbins=True, labels=False)
            df['gen_'+col+'_bin_'+str(binnums)]=pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
            df_same[col]=pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
        except:
            print('err',col)
    return df,df_same

