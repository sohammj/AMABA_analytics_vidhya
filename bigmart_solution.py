
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")

path = sys.argv[1] if len(sys.argv) > 1 else "."
out = sys.argv[2] if len(sys.argv) > 2 else "bigmart_submission.csv"

a = pd.read_csv(path + "/train_v9rqX0R.csv")
b = pd.read_csv(path + "/test_AbJTz2l.csv")

# Data cleaning
a["is_train"] = 1
b["is_train"] = 0

c = pd.concat([a, b], ignore_index=True)

# Fix different names used for fat content
c["Item_Fat_Content"] = c["Item_Fat_Content"].replace({
    "LF": "Low Fat",
    "low fat": "Low Fat",
    "reg": "Regular"
})

# NC items are non-edible
c.loc[c["Item_Identifier"].str[:2] == "NC", "Item_Fat_Content"] = "Non-Edible"

# Fill missing item weights
c["Item_Weight"] = c["Item_Weight"].fillna(
    c.groupby("Item_Identifier")["Item_Weight"].transform("mean")
)

c["Item_Weight"] = c["Item_Weight"].fillna(
    c.groupby("Item_Type")["Item_Weight"].transform("mean")
)

# Replace zero visibility with average visibility
v = c["Item_Visibility"].replace(0, np.nan)

c["Item_Visibility"] = v.fillna(
    v.groupby(c["Item_Identifier"]).transform("mean")
).fillna(v.mean())

# Create extra features
c["Outlet_Age"] = 2013 - c["Outlet_Establishment_Year"]

c["Item_cnt"] = c.groupby("Item_Identifier")["Item_Identifier"].transform("size")

# Separate train and test again
x = c[c["is_train"] == 1].copy().reset_index(drop=True)
z = c[c["is_train"] == 0].copy().reset_index(drop=True)

y = x["Item_Outlet_Sales"].values

# Convert categories into numbers for LightGBM
for d in [x, z]:
    d["Outlet_code"] = d["Outlet_Identifier"].str[-3:].astype(int)
    d["Type_code"] = d["Item_Type"].astype("category").cat.codes

z["Type_code"] = pd.Categorical(
    z["Item_Type"],
    categories=x["Item_Type"].astype("category").cat.categories
).codes

outlets = sorted(x["Outlet_Identifier"].unique())


# Model 1: Ridge Regression
def ridge_model(a, y, b):

    def make_features(d):
        p = np.column_stack([
            (d["Outlet_Identifier"] == o).astype(float)
            for o in outlets
        ])

        q = d["Item_MRP"].values[:, None] / 100

        return np.hstack([p, p * q, p * q ** 2])

    model = Ridge(alpha=1.0)
    model.fit(make_features(a), y)

    return model.predict(make_features(b))


# Model 2: CatBoost
cat_cols = [
    "Item_MRP",
    "Outlet_Identifier",
    "Item_Type",
    "Item_Fat_Content",
    "Item_Visibility",
    "Item_Weight",
    "Outlet_Age"
]

def cat_model(a, y, b):
    p = np.zeros(len(b))

    for s in [0, 1, 2]:
        model = CatBoostRegressor(
            iterations=600,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=10,
            random_seed=s,
            cat_features=[1, 2, 3],
            verbose=0,
            thread_count=-1
        )

        model.fit(a[cat_cols], y)
        p += model.predict(b[cat_cols]) / 3

    return p


# Model 3: LightGBM
lgb_cols = [
    "Item_MRP",
    "Outlet_code",
    "Item_Visibility",
    "Item_Weight",
    "Type_code",
    "Outlet_Age",
    "Item_cnt"
]

def lgb_model(a, y, b):
    p = np.zeros(len(b))

    for s in [0, 1, 2]:
        model = lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.02,
            num_leaves=6,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            reg_lambda=10,
            random_state=s,
            verbose=-1
        )

        model.fit(a[lgb_cols], y)
        p += model.predict(b[lgb_cols]) / 3

    return p


# Store the models and their weights
models = {
    "Ridge": ridge_model,
    "CatBoost": cat_model,
    "LightGBM": lgb_model
}

weights = {
    "Ridge": 0.5,
    "CatBoost": 0.3,
    "LightGBM": 0.2
}


# Calculate RMSE
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


# 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv = {}

for name in models:
    cv[name] = np.zeros(len(x))

for i, j in kf.split(x):
    a1 = x.iloc[i]
    b1 = x.iloc[j]
    y1 = y[i]

    for name, fn in models.items():
        cv[name][j] = fn(a1, y1, b1)


# Check individual model scores
for name in models:
    score = rmse(y, cv[name])
    print(name, "RMSE:", round(score, 2))


# Combine predictions from all three models
p = np.zeros(len(x))

for name in models:
    p += weights[name] * cv[name]

print("Final CV RMSE:", round(rmse(y, p), 2))


# Train using the complete training data
pred = np.zeros(len(z))

for name, fn in models.items():
    pred += weights[name] * fn(x, y, z)

# Sales cannot be below the minimum observed sales
pred = np.clip(pred, 33.29, None)


# Save the final submission
result = pd.DataFrame({
    "Item_Identifier": z["Item_Identifier"],
    "Outlet_Identifier": z["Outlet_Identifier"],
    "Item_Outlet_Sales": pred
})

result.to_csv(out, index=False)

print("Submission saved:", out)
print("Shape:", result.shape)
