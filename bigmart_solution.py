"""
Big Mart Sales Prediction - Analytics Vidhya
Approach: light feature engineering + blend of
  (1) Ridge with Outlet x Item_MRP (poly-2) interactions,
  (2) CatBoost (categorical outlet / item type),
  (3) LightGBM (shallow, heavily regularised).
Weights were chosen from 5-fold CV out-of-fold predictions.
Run:  python bigmart_solution.py [data_dir] [out_csv]
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

D = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bigmart_submission.csv"
SEED = 42
train = pd.read_csv(D + "train_v9rqX0R.csv")
test = pd.read_csv(D + "test_AbJTz2l.csv")

def build(train, test):
    train = train.copy(); test = test.copy()
    train["is_train"] = 1; test["is_train"] = 0
    df = pd.concat([train, test], ignore_index=True)
    df["Item_Fat_Content"] = df.Item_Fat_Content.replace({"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})
    df.loc[df.Item_Identifier.str[:2] == "NC", "Item_Fat_Content"] = "Non-Edible"
    # Item_Weight: same item -> same weight
    df["Item_Weight"] = df.Item_Weight.fillna(df.groupby("Item_Identifier").Item_Weight.transform("mean"))
    df["Item_Weight"] = df.Item_Weight.fillna(df.groupby("Item_Type").Item_Weight.transform("mean"))
    # zero visibility is a missing value
    v = df.Item_Visibility.replace(0, np.nan)
    df["Item_Visibility"] = v.fillna(v.groupby(df.Item_Identifier).transform("mean")).fillna(v.mean())
    df["Outlet_Age"] = 2013 - df.Outlet_Establishment_Year
    df["Item_cnt"] = df.groupby("Item_Identifier").Item_Identifier.transform("size")
    return df

df = build(train, test)
T = df[df.is_train == 1].reset_index(drop=True)
S = df[df.is_train == 0].reset_index(drop=True)
y = T.Item_Outlet_Sales.values
for d in (T, S):
    d["Outlet_code"] = d.Outlet_Identifier.str[-3:].astype(int)
    d["Type_code"] = d.Item_Type.astype("category").cat.codes
S["Type_code"] = pd.Categorical(S.Item_Type, categories=T.Item_Type.astype("category").cat.categories).codes
outlets = sorted(T.Outlet_Identifier.unique())

# ---------------- models ----------------
def ridge_fit_predict(A, ya, B):
    def M(d):
        X = np.column_stack([(d.Outlet_Identifier == o).astype(float) for o in outlets])
        m = d.Item_MRP.values[:, None] / 100
        return np.hstack([X, X * m, X * m ** 2])
    return Ridge(alpha=1.0).fit(M(A), ya).predict(M(B))

CB_F = ["Item_MRP", "Outlet_Identifier", "Item_Type", "Item_Fat_Content", "Item_Visibility", "Item_Weight", "Outlet_Age"]
def cat_fit_predict(A, ya, B, seeds=(0, 1, 2)):
    p = 0
    for s in seeds:
        m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10, verbose=0,
                              random_seed=s, thread_count=-1, cat_features=[1, 2, 3])
        p += m.fit(A[CB_F], ya).predict(B[CB_F]) / len(seeds)
    return p

LG_F = ["Item_MRP", "Outlet_code", "Item_Visibility", "Item_Weight", "Type_code", "Outlet_Age", "Item_cnt"]
def lgb_fit_predict(A, ya, B, seeds=(0, 1, 2)):
    p = 0
    for s in seeds:
        m = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.02, num_leaves=6, min_child_samples=60,
                              subsample=0.8, subsample_freq=1, reg_lambda=10, random_state=s, verbose=-1)
        p += m.fit(A[LG_F], ya).predict(B[LG_F]) / len(seeds)
    return p

MODELS = {"ridge": ridge_fit_predict, "catboost": cat_fit_predict, "lgbm": lgb_fit_predict}
W = {"ridge": 0.5, "catboost": 0.3, "lgbm": 0.2}   # from CV OOF analysis

# ---------------- 5-fold CV (validation only; NOT the leaderboard score) ----------------
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
oof = {k: np.zeros(len(T)) for k in MODELS}
for tr_i, va_i in KFold(5, shuffle=True, random_state=SEED).split(T):
    for k, f in MODELS.items():
        oof[k][va_i] = f(T.iloc[tr_i], y[tr_i], T.iloc[va_i])
for k in MODELS: print(f"CV RMSE {k:9s}: {rmse(oof[k], y):.2f}")
blend = sum(W[k] * oof[k] for k in MODELS)
print(f"CV RMSE blend    : {rmse(blend, y):.2f}")

# ---------------- fit on all training data, predict test ----------------
pred = sum(W[k] * MODELS[k](T, y, S) for k in MODELS)
pred = np.clip(pred, 33.29, None)   # min observed sales in train
sub = pd.DataFrame({"Item_Identifier": S.Item_Identifier, "Outlet_Identifier": S.Outlet_Identifier,
                    "Item_Outlet_Sales": pred})
sub.to_csv(OUT, index=False)
print("saved", OUT, sub.shape)
