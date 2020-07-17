import pandas as pd
import wandb
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import argparse
import os

def main(args):

    branch = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()
    os.system("git commit -am \"{0}\"".format(args.message))
    wandb.init( project = "kalapa")
    config = {
        "data_version" : args.data_version,
        "code_version" : args.code_version,
        "weight_version" : args.weight_version
    }

    df_train = pd.read_csv(f"../../data/kalapa/{config['data_version']}/train.csv")
    df_dev = pd.read_csv(f"../../data/kalapa/{config['data_version']}/dev.csv")
    #df_test = pd.read_csv(f"../../data/kalapa/{config['data_version']}/test.csv")

    X_train = df_train.iloc[:,1:]
    y_train = df_train["label"]

    X_dev = df_dev.iloc[:,1:]
    y_dev = df_dev["label"]

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_dev)
    predictions = predictions[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(y_dev, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    ginicof = 2*auc - 1

    config['gini'] = ginicof
    wandb.log(config)

    """X_transform = [df_train, df_dev]
    X_transform = pd.concat(X_transform)

    X = X_transform.iloc[:,1:]
    y = X_transform.iloc[:,0]
    model = GradientBoostingClassifier()
    model.fit(X,y)

    test = df_test.iloc[:,1:]
    test_id = df_test["label"]
    predictions = model.predict_proba(test)
    predictions = predictions[:,1]
    f = open(f"../../data/kalapa/{config['data_version']}/submission.csv", "w")
    f.write("id,label\n")
    for i in range(len(predictions)):
        f.write(str(test_id.iloc[i]) + "," + str(predictions[i]) + "\n")
    f.close()"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--code-version", type=str)
    parser.add_argument("-m", "--message", type=str)
    parser.add_argument("-w", "--weight-version", type=str, default="")
    parser.add_argument("-d", "--data-version", type=str)
    args = parser.parse_args()
    main(args)