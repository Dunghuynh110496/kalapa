import pandas as pd
from sklearn import metrics
import wandb
import os
import argparse
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def compute_gini(labels, preds):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    gini = 2 * auc - 1
    return gini

def main(args):
    wandb.init(project="kalapa")
    seed = args.seed
    os.system(f"git commit -am \"{args.message}\"")
    code_version = os.popen('git rev-parse HEAD').read().strip()
    wandb.log({"user": "cho",
               "seed": seed,
               "code_version": code_version,
               "data_version": args.data_version,
               "weight_version": args.weight_version})

    train_dev = pd.read_csv(f"../../data/kalapa/{args.data_version}/train.csv")
    test = pd.read_csv(f"../../data/kalapa/{args.data_version}/test.csv")
    train, dev = train_test_split(train_dev, test_size=0.08, stratify=train_dev.label, random_state=10)

    best_gini = -1.0
    best_dev_pred = None
    best_test_pred = None
    wandb.log({"gini": best_gini})
    X_train = train[:,1:]
    y_train = train[:,0]
    X_dev = dev[:,1:]
    y_dev = dev[:,0]

    clf = lgb.train(X_train,y_train)
    predictions = clf.predict_proba(X_dev)
    predictions = predictions[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_dev, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    ginicof = 2 * auc - 1
    print("gini xgb" + str(ginicof))

    """dev["pred"] = best_dev_pred
    test["label"] = best_test_pred
    dev[["id", "label", "pred"]].to_csv("dev_preds.csv", index=False)
    test[["id", "label"]].to_csv("test_preds.csv", index=False)
    wandb.save("dev_preds.csv")
    wandb.save("test_preds.csv")"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", type=str)
    parser.add_argument("-w", "--weight-version", type=str, default="")
    parser.add_argument("-d", "--data-version", type=str)
    parser.add_argument("-s", "--seed", type=int, default=10)
    args = parser.parse_args()
    main(args)