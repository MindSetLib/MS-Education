
# создание групп по бинам
def create_bin_groups(df,binnums=100):
    for col in df.columns:
        try:
            ser, bins = pd.qcut(df[col].values, binnums,duplicates ='drop', retbins=True, labels=False)
            df['gen_'+col+'_bin_'+str(binnums)]=pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
        except:
            print('err',col)
    return df


