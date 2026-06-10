import pandas as pd
import json

def load_config(config_path):
  with open(config_path, 'r') as file:
    return json.load(file)

def load_data(data_path, config):
  df = pd.read_csv(data_path, comment="#")

  columns = config["feature_columns"] + [config["label_column"]]
  df = df[columns]

  df = df.dropna()

  label_map = config["label_map"]
  df[config["label_column"]] = df[config["label_column"]].map(label_map)

  df = df.dropna(subset = [config["label_column"]])

  return df