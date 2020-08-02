import pandas as pd
from sklearn import metrics
import wandb
import os
import argparse
import lightgbm as lgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def gini(y_true, y_score):
    return roc_auc_score(y_true, y_score)*2 - 1
def lgb_gini(y_pred, dataset_true):
    y_true = dataset_true.get_label()
    return 'gini', gini(y_true, y_pred), True


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
    def cate(df_fe):
        for col in df_fe.columns:
            if df_fe[col].dtype.name == "object":
                df_fe[col] = df_fe[col].astype('category')
        return df_fe
    train = cate(train)
    test = cate(test)
    print(train.columns)

    lgbm_param = {'boosting_type': 'gbdt', \
                  'colsample_bytree': 0.6602479798930369, \
                  'is_unbalance': False, \
                  'learning_rate': 0.00746275526696824, \
                  'max_depth': 15, \
                  'metric': 'auc', \
                  'min_child_samples': 25, \
                  'num_leaves': 60, \
                  'objective': 'binary', \
                  'reg_alpha': 0.4693391197064131, \
                  'reg_lambda': 0.16175478669541327, \
                  'subsample_for_bin': 60000}
    NUM_BOOST_ROUND = 10000

    def kfold(train_fe, test_fe):
        y_label = train_fe.label
        seeds = np.random.randint(0, 10000, 1)
        preds = 0
        feature_important = None
        avg_train_gini = 0
        avg_val_gini = 0

        for s in seeds:
            skf = StratifiedKFold(n_splits=5, random_state=6484, shuffle=True)
            lgbm_param['random_state'] = 6484
            seed_train_gini = 0
            seed_val_gini = 0
            for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_label)), y_label)):
                X_train, X_val = train_fe.iloc[train_idx].drop(["id", "label"], 1), train_fe.iloc[val_idx].drop(
                    ["id", "label"], 1)
                print(X_train)
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]

                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_val, y_val)

                evals_result = {}
                model = lgb.train(lgbm_param,
                                  lgb_train,
                                  num_boost_round=NUM_BOOST_ROUND,
                                  early_stopping_rounds=400,
                                  feval=lgb_gini,
                                  verbose_eval=200,
                                  evals_result=evals_result,
                                  valid_sets=[lgb_train, lgb_eval])

                seed_train_gini += model.best_score["training"]["gini"] / skf.n_splits
                seed_val_gini += model.best_score["valid_1"]["gini"] / skf.n_splits

                avg_train_gini += model.best_score["training"]["gini"] / (len(seeds) * skf.n_splits)
                avg_val_gini += model.best_score["valid_1"]["gini"] / (len(seeds) * skf.n_splits)

                log = {
                    "gini_train": avg_train_gini,
                    "gini": avg_val_gini
                }
                wandb.log(log)
                if feature_important is None:
                    feature_important = model.feature_importance() / (len(seeds) * skf.n_splits)
                else:
                    feature_important += model.feature_importance() / (len(seeds) * skf.n_splits)

                pred = model.predict(test_fe.drop(["id"], 1))
                preds += pred / (skf.n_splits * len(seeds))

                print("Fold {}: {}/{}".format(i, model.best_score["training"]["gini"],
                                              model.best_score["valid_1"]["gini"]))
            print("Seed {}: {}/{}".format(s, seed_train_gini, seed_val_gini))

        print("-" * 30)
        print("Avg train gini: {}".format(avg_train_gini))
        print("Avg valid gini: {}".format(avg_val_gini))
        print("=" * 30)
        return preds
    preds  = kfold(train, test)
    test["label"] = preds
    test[["id", "label"]].to_csv("test_preds.csv", index=False)
    wandb.save("test_preds.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", type=str)
    parser.add_argument("-w", "--weight-version", type=str, default="")
    parser.add_argument("-d", "--data-version", type=str)
    parser.add_argument("-s", "--seed", type=int, default=10)
    args = parser.parse_args()
    main(args)