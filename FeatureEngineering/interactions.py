
# полиномиальные фичи

>>> import numpy as np
>>> from sklearn.preprocessing import PolynomialFeatures
>>> X = np.arange(6).reshape(3, 2)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> poly = PolynomialFeatures(2)
>>> poly.fit_transform(X)
array([[ 1.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  2.,  3.,  4.,  6.,  9.],
       [ 1.,  4.,  5., 16., 20., 25.]])
>>> poly = PolynomialFeatures(interaction_only=True)
>>> poly.fit_transform(X)
array([[ 1.,  0.,  1.,  0.],
       [ 1.,  2.,  3.,  6.],
       [ 1.,  4.,  5., 20.]])



#gaussian features

>>> import numpy as np
>>> from sklearn.preprocessing import PowerTransformer
>>> pt = PowerTransformer()
>>> data = [[1, 2], [3, 2], [4, 5]]
>>> print(pt.fit(data))
PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
>>> print(pt.lambdas_)
[ 1.386... -3.100...]
>>> print(pt.transform(data))
[[-1.316... -0.707...]
 [ 0.209... -0.707...]
 [ 1.106...  1.414...]]


#Robust scaler
>>> from sklearn.preprocessing import RobustScaler
>>> X = [[ 1., -2.,  2.],
...      [ -2.,  1.,  3.],
...      [ 4.,  1., -2.]]
>>> transformer = RobustScaler().fit(X)
>>> transformer
RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
       with_scaling=True)
>>> transformer.transform(X)
array([[ 0. , -2. ,  0. ],
       [-1. ,  0. ,  0.4],
       [ 1. ,  0. , -1.6]])

#Power transform
>>> import numpy as np
>>> from sklearn.preprocessing import power_transform
>>> data = [[1, 2], [3, 2], [4, 5]]
>>> print(power_transform(data, method='box-cox'))
[[-1.332... -0.707...]
 [ 0.256... -0.707...]
 [ 1.076...  1.414...]]


#функция взаимодействий

from itertools import combinations

def interactions(data):
    columns=list(data.columns)
    ls=list(combinations(columns, 2))
    for inter in ls:
        print(inter[0], inter[1])
        data[str(inter[0])+'_'+str(inter[1])]=data[str(inter[0])]+data[str(inter[1])]
        data[str(inter[0])+'*'+str(inter[1])]=data[str(inter[0])]*data[str(inter[1])]
        data[str(inter[0])+'/'+str(inter[1])]=data[str(inter[0])]/data[str(inter[1])]
        data[str(inter[0])+'-'+str(inter[1])]=data[str(inter[0])]/data[str(inter[1])]
        data[str(inter[0])+'x2']=data[str(inter[0])]*data[str(inter[0])]
        data[str(inter[0])+'log']=data[str(inter[0])].apply(np.log)
        data[str(inter[0])+'-s-'+str(inter[1])]=data[str(inter[0])].astype(str)+data[str(inter[1])].astype(str)  
        le = preprocessing.LabelEncoder()
        le.fit(data[str(inter[0])+'-s-'+str(inter[1])])
        data[str(inter[0])+'-s-'+str(inter[1])]=le.transform(data[str(inter[0])+'-s-'+str(inter[1])])
    return data

# логарифмы

def log_features(data,ttype):
    categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
    numerical_columns   = [c for c in data.columns if (data[c].dtype.name != 'object') and (data[c].dtype.name != 'datetime64[ns]') and (data[c].dtype.name != 'category')]
    for ncol in numerical_columns:
        data[str(ncol)+'_'+str(ttype)]=(data[ncol]).apply(np.log)
    return data