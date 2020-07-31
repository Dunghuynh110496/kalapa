import pandas as pd
from sklearn import metrics
import wandb
import os
import argparse
import lightgbm as lgb
import numpy as np

from sklearn.model_selection import KFold

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

    train_dev = pd.read_csv(f"../../data/kalapa/{args.data_version}/train.csv")
    test = pd.read_csv(f"../../data/kalapa/{args.data_version}/test.csv")
    new_data = pd.read_csv(f"../../data/kalapa/{args.data_version}/new_data.csv")
    train_dev = pd.concat([train_dev,new_data ], axis = 0)
    train_dev.head()
    params = {}
    params['learning_rate'] = 0.01
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.1
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    params['seed'] = seed
    def ginicof(y, preds):
        fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        ginicof = 2 * auc - 1
        return ginicof
    def evaluate(x):
        if x.iteration % 100 == 0:
            nonlocal best_gini, best_dev_pred, best_test_pred, i
            predictions_dev = x.model.predict(dev)
            predictions_train = x.model.predict(train)
            predictions_test = x.model.predict(test.iloc[:, 1:])
            gini_dev = ginicof(train_dev.iloc[dev_index]["label"], predictions_dev)
            if gini_dev > best_gini:
                wandb.log({"gini": best_gini})
                best_gini = gini_dev
                best_dev_pred = predictions_dev
                best_test_pred = predictions_test
            gini_train = ginicof(train_dev.iloc[train_index]["label"], predictions_train)
            log_iter = {"gini_dev": gini_dev,
                   "gini_train": gini_train,
                   "gini" : best_gini,
                   "epoch": x.iteration,
                    "iter": i}
            print(log_iter)


    iter_pred_dev = []
    iter_pred_test = []
    ginis = []

    for i in range(1):
        pred_dev_stack = []
        pred_test_stack = []
        kf = KFold(n_splits = 8, shuffle=True)
        fold = kf.split(train_dev)
        for train_index, dev_index in fold:
            best_gini = -1.0
            best_dev_pred = None
            best_test_pred = None

            train = train_dev.iloc[train_index, 2:]
            dev = train_dev.iloc[dev_index, 2:]

            d_train = lgb.Dataset(train, label=train_dev.iloc[train_index]["label"])
            clf = lgb.train(params,
                      d_train,
                      2500,
                    callbacks=[evaluate])
            ginis.append(best_gini)
            pred_dev_stack.append(best_dev_pred)
            pred_test_stack.append(best_test_pred)
        iter_pred_dev.append(pred_dev_stack)
        iter_pred_test.append(pred_test_stack)
    gini = np.mean(np.array(ginis))
    predictions_dev = np.asarray(iter_pred_dev)
    predictions_dev = np.mean(predictions_dev, axis=0)
    predictions_test = np.asarray(iter_pred_test)
    predictions_test = np.mean(predictions_test, axis=0)
    log = {
       "gini" : gini
    }

    wandb.log(log)
    dev["pred"] = predictions_dev
    test["label"] = predictions_test
    dev[["id", "label", "pred"]].to_csv("dev_preds.csv", index=False)
    test[["id", "label"]].to_csv("test_preds.csv", index=False)
    wandb.save("dev_preds.csv")
    wandb.save("test_preds.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", type=str)
    parser.add_argument("-w", "--weight-version", type=str, default="")
    parser.add_argument("-d", "--data-version", type=str)
    parser.add_argument("-s", "--seed", type=int, default=10)
    args = parser.parse_args()
    main(args)