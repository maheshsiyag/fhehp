import shutil

from pathlib import Path

import pandas as pd

from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.deployment import FHEModelDev

# Files location
TRAINING_FILE_NAME = "./data/Training_preprocessed.csv"
TESTING_FILE_NAME = "./data/Testing_preprocessed.csv"

# Load data
df_train = pd.read_csv(TRAINING_FILE_NAME)
df_test = pd.read_csv(TESTING_FILE_NAME)

print(df_train.shape)
print(df_train.columns)
# Split the data into X_train, y_train, X_test_, y_test sets
TARGET_COLUMN = ["prognosis_encoded", "prognosis"]

y_train = df_train[TARGET_COLUMN[0]].values.flatten()
y_test = df_test[TARGET_COLUMN[0]].values.flatten()

X_train = df_train.drop(TARGET_COLUMN, axis=1)
X_test = df_test.drop(TARGET_COLUMN, axis=1)

# Models parameters
optimal_param = {"C": 0.9, "n_bits": 13, "solver": "sag", "multi_class": "auto"}

# Concrete ML model
clf = ConcreteLogisticRegression(**optimal_param)

clf.fit(X_train, y_train)

fhe_circuit = clf.compile(X_train)

fhe_circuit.client.keygen(force=False)

path_to_model = Path("./deployment_logit/").resolve()

if path_to_model.exists():
    shutil.rmtree(path_to_model)

dev = FHEModelDev(path_to_model, clf)

dev.save(via_mlir=True)