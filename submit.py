import pandas as pd
import wandb
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

wandb.init( project = "kalapa")
config = {
    "data_version" : "12-07-2020-test",
    "code_version" : "",
    "weight_version" : ""
}


df_train = pd.read_csv(f"../../data/kalapa/{config['data_version']}/train.csv")
df_dev = pd.read_csv(f"../../data/kalapa/{config['data_version']}/dev.csv")
test = pd.read_csv(f"../../data/kalapa/{config['data_version']}/test.csv")

X_train = df_train.iloc[:,1:]
y_train = df_train["label"]

X_dev = df_dev.iloc[:,1:]
y_dev = df_dev["label"]

model_xgb = GradientBoostingClassifier(random_state = 50)
model_xgb.fit(X_train, y_train)
predictions_xgb = model_xgb.predict_proba(X_dev)
predictions_xgb = predictions_xgb[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_dev, predictions_xgb, pos_label=1)
auc_xgb = metrics.auc(fpr, tpr)
gini_xgb = 2*auc_xgb - 1

config['gini'] = gini_xgb
wandb.log(config)

predictions_xgb = model_xgb.predict_proba(test)
predictions_xgb = predictions_xgb[:,1]

print(predictions_xgb)

