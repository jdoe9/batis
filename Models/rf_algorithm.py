import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import json
import ast

bioclimatic_raster_names = [
    "bio_1", "bio_2", "bio_3", "bio_4", "bio_5", "bio_6", "bio_7", "bio_8", "bio_9",
    "bio_10", "bio_11", "bio_12", "bio_13", "bio_14", "bio_15", "bio_16", "bio_17",
    "bio_18", "bio_19"
]

RANDOM_STATE    = 581
RF_TREES  = 100

def main():
    df_train = pd.read_csv("finalized_splits_kenya/train_filtered.csv")
    df_test = pd.read_csv("finalized_splits_kenya/test_filtered.csv")

    target_vectors = []

    X   = df_train[bioclimatic_raster_names].astype(np.float32).to_numpy()
    df_train["target"] = df_train["target"].apply(ast.literal_eval)
    y = np.array(list(df_train['target']))


    X_test = df_test[bioclimatic_raster_names].astype(np.float32).to_numpy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf_bounded",
         TransformedTargetRegressor(
             regressor=RandomForestRegressor(
                 n_estimators=RF_TREES,
                 n_jobs=-1,
                 random_state=RANDOM_STATE
             )
         )
        )
    ])

    model.fit(X, y)

    preds = model.predict(X_test) 
    y_pred_scaled = np.clip(preds, 0, 1)

    for i in range(len(df_test)):
        hotspot = df_test.iloc[i]['hotspot_id']
        print(hotspot)
        np.save(f"/Users/Testing_Env/evaluate_results/predictions_kenya/rf/3/{hotspot}.npy", preds[i])

if __name__ == "__main__":
    main()