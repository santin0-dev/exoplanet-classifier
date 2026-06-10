import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(df, config):
  feature_columns = config["feature_columns"]

  X = df[feature_columns].values
  y = df[config["label_column"]].values

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  return X_scaled, y, scaler