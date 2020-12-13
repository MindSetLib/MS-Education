train_predictions=pd.DataFrame(lgb_model.predict(train_f[features_columns]))
train_predictions=train_predictions.rename(columns={0:'isFraud_pred'})
train_predictions[TARGET]=train_f[TARGET]
train_predictions.shape

########################### Export
if not LOCAL_TEST:
    test_predictions['isFraud'] = test_predictions['prediction']
    test_predictions[['TransactionID','isFraud']].to_csv('experiment04092019_1/test_pred_04092019_1.csv', index=False)
    train_predictions.to_csv('experiment04092019_1/train_pred_04092019_1.csv', index=False)