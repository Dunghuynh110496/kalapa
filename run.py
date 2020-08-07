import pandas as pd
import wandb
import os
import argparse

import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

def gini(y_true, y_score):
    return roc_auc_score(y_true, y_score)*2 - 1

def evaluate(i,model, X_train, y_train, X_dev, y_dev, test):
    train_preds = model.predict(X_train)
    dev_preds = model.predict(X_dev)

    test_preds = model.predict_proba(test)[:,1]
    train_proba = model.predict_proba(X_train)[:,1]
    dev_proba = model.predict_proba(X_dev)[:,1]

    train_errors = abs(train_preds - y_train)
    mape = 100 * np.mean(train_errors / y_train)
    train_accuracy = 100 - mape

    dev_errors = abs(dev_preds - y_dev)
    mape = 100 * np.mean(dev_errors / y_dev)
    dev_accuracy = 100 - mape

    train_gini = gini(y_train, train_proba)
    dev_gini = gini(y_dev, dev_proba)

    log1 = {
        "gini_train": train_gini,
        "gini": dev_gini
    }
    wandb.log(log1)

    print('Model Performance')
    print(f"fold: {i}, train_gini: {train_gini}, dev_gini: {dev_gini}" )

    return [test_preds, train_gini, dev_gini]

def main(args):
    wandb.init(project="kalapa")
    seed = args.seed
    os.system(f"git commit -am \"{args.message}\"")
    code_version = os.popen('git rev-parse HEAD').read().strip()
    wandb.log({"user": "parker",
               "seed": seed,
               "code_version": code_version,
               "data_version": args.data_version,
               "weight_version": args.weight_version})
    train = pd.read_csv(f"../../data/kalapa/{args.data_version}/train.csv")
    test = pd.read_csv(f"../../data/kalapa/{args.data_version}/test.csv")

    clf = RandomForestClassifier(
        n_estimators=50,
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=0,
        verbose=0,
        warm_start=False,
        class_weight='balanced'
    )
    def kfold(train_fe, test_fe):
        y_label = train_fe.label
        test_preds = 0
        avg_train_gini = 0
        avg_dev_gini = 0
        skf = StratifiedKFold(n_splits=5, random_state=6484, shuffle=True)

        for i, (train_idx, dev_idx) in enumerate(skf.split(np.zeros(len(y_label)), y_label)):
            X_train = train_fe.iloc[train_idx].drop(["id", "label"], 1)
            X_dev = train_fe.iloc[dev_idx].drop(["id", "label"], 1)
            y_train = y_label.iloc[train_idx]
            y_dev = y_label.iloc[dev_idx]
            X_test = test_fe.iloc[:,1:]
            clf.fit(X_train, y_train)
            #output =  [test_preds, train_gini, dev_gini]
            output = evaluate(i,clf, X_train, y_train, X_dev, y_dev, X_test)

            test_pred = output[0]
            train_gini = output[1]
            dev_gini = output[2]

            test_preds += test_pred / (skf.n_splits)
            avg_train_gini += train_gini/ (skf.n_splits)
            avg_dev_gini += dev_gini/ (skf.n_splits)


        print("-" * 30)
        print("Avg train gini: {}".format(avg_train_gini))
        print("Avg valid gini: {}".format(avg_dev_gini))
        print("=" * 30)
        log2 = {
            "gini": avg_dev_gini
        }
        wandb.log(log2)
        return test_preds
    preds = kfold(train, test)
    print(preds)
    test["label"] = preds
    print("a")
    test[["id", "label"]].to_csv("test_preds_rf.csv", index=False)
    wandb.save("test_preds_rf.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", type=str)
    parser.add_argument("-w", "--weight-version", type=str, default="")
    parser.add_argument("-d", "--data-version", type=str)
    parser.add_argument("-s", "--seed", type=int, default=10)
    args = parser.parse_args()
    main(args)