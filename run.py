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
    wandb.log({"user": "parker",
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
    d_train = lgb.Dataset(train.iloc[:, 2:], label=train.label)
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
            nonlocal best_gini, best_dev_pred, best_test_pred
            predictions_dev = x.model.predict(dev.iloc[:,2:])
            predictions_train = x.model.predict(train.iloc[:,2:])
            predictions_test = x.model.predict(test.iloc[:, 1:])
            gini_dev = ginicof(dev.iloc[:,1], predictions_dev)
            if gini_dev > best_gini:
                best_gini = gini_dev
                best_dev_pred = predictions_dev
                best_test_pred = predictions_test
            gini_train = ginicof(train.iloc[:,1], predictions_train)
            log = {"gini_dev": gini_dev,
                   "gini_train": gini_train,
                   "gini" : best_gini,
                   "epoch": x.iteration}
            print(log)
            wandb.log(log)
    clf = lgb.train(params,
              d_train,
              ,
            callbacks=[evaluate])
    dev["pred"] = best_dev_pred
    test["label"] = best_test_pred
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