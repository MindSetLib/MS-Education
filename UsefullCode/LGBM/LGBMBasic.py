########################### Model
import lightgbm as lgb


def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    X, y = tr_df[features_columns], tr_df[target]
    P, P_y = tt_df[features_columns], tt_df[target]

    tt_df = tt_df[['TransactionID', target]]
    predictions = np.zeros(len(tt_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold:', fold_)
        tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]

        print(len(tr_x), len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        if LOCAL_TEST:
            vl_data = lgb.Dataset(P, label=P_y)
        else:
            vl_data = lgb.Dataset(vl_x, label=vl_y)

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets=[tr_data, vl_data],
            verbose_eval=200,
        )

        pp_p = estimator.predict(P)
        predictions += pp_p / NFOLDS

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), X.columns)),
                                       columns=['Value', 'Feature'])
            print(feature_imp)

        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()

    tt_df['prediction'] = predictions

    return tt_df, estimator
## -------------------



lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.012,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.5,
                    'subsample_freq':1,
                    'subsample':0.5,
                    'n_estimators':1500,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100,
                }


########################### Model Train
#if LOCAL_TEST:
#    lgb_params['learning_rate'] = 0.01
#    lgb_params['n_estimators'] = 20000
#    lgb_params['early_stopping_rounds'] = 100
#    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)
#    print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
#else:
lgb_params['learning_rate'] = 0.012
lgb_params['n_estimators'] = 1500
lgb_params['early_stopping_rounds'] = 100
%time test_predictions,lgb_model = make_predictions(train_f, test_f, features_columns, TARGET, lgb_params, NFOLDS=3)