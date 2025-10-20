import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"D:\Genai_Projects\ai_ecom_agent\Data\churn.csv")


X = df.drop(columns=["churn","profit_margin"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.33, random_state=42,shuffle=True,stratify=y)


one_cols = ["category", "payment_method", "region"]
ord_cols = ["returned","customer_gender"]
num_cols = ["price","discount_percentage","quantity","delivery_time_days","total_amount","shipping_cost","customer_age","date","month","year"]

one_pipeline = Pipeline([
    ("onehotencoder",OneHotEncoder())
])

ordi_pipeline = Pipeline([
    ("ordinalencoder",OrdinalEncoder())
])

scaler_pipeline = Pipeline([
    ("scaler",StandardScaler())
])
ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=ratio, eval_metric="auc", random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[   
        ("onehot", one_pipeline,    one_cols),
        ("ord",    ordi_pipeline,   ord_cols),
        ("num",    scaler_pipeline, num_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb)
])


pipeline.fit(X_train, y_train)

with open(r"D:\Genai_Projects\ai_ecom_agent\models\churn_xgb_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Saved: churn_xgb_pipeline.pkl")


