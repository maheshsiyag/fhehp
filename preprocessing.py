"""
Preliminary preprocessing on the data, such as:
- correcting column names
- encoding the target column
"""

import pandas as pd
from sklearn import preprocessing

COLUMNS_TO_DROP = ["Unnamed: 133"]
TARGET_COLUMN = ["prognosis"]
RENAME_COLUMNS = {
    "scurring": "scurving",
    "dischromic _patches": "dischromic_patches",
    "spotting_ urination": "spotting_urination",
    "foul_smell_of urine": "foul_smell_of_urine",
}


def pretty_print(input):
    """
    Prettify the input.

    Args:
        input: Can be a list of symtoms or a disease.

    Returns:
        list: Sorted and prettified input.
    """
    # Convert to a list if necessary
    if isinstance(input, list):
        input = list(input)

    # Flatten the list if required
    pretty_list = []
    for item in input:
        if isinstance(item, list):
            pretty_list.extend(item)
        else:
            pretty_list.append(item)

    # Sort and prettify the input
    pretty_list = sorted([" ".join((item.split("_"))).title() for item in pretty_list])

    return pretty_list


def map_prediction(target_columns=["y", "prognosis"]):
    df = pd.read_csv("Training_preprocessed.csv")
    relevent_df = df[target_columns].drop_duplicates().relevent_df.where(df["y"] == 1)
    prediction = relevent_df[target_columns[1]].dropna().values[0]
    return prediction


if __name__ == "__main__":

    # Load data
    df_train = pd.read_csv("Training.csv")
    df_test = pd.read_csv("Testing.csv")

    # Remove unseless columns
    df_train.drop(columns=COLUMNS_TO_DROP, axis=1, errors="ignore", inplace=True)
    df_test.drop(columns=COLUMNS_TO_DROP, axis=1, errors="ignore", inplace=True)

    # Correct some typos in some columns name
    df_train.rename(columns=RENAME_COLUMNS, inplace=True)
    df_test.rename(columns=RENAME_COLUMNS, inplace=True)

    # Convert y category labels to y
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df_train[TARGET_COLUMN].values.flatten())

    df_train["y"] = label_encoder.transform(df_train[TARGET_COLUMN].values.flatten())
    df_test["y"] = label_encoder.transform(df_test[TARGET_COLUMN].values.flatten())

    # Cast X features from int64 to float32
    float_columns = df_train.columns.drop(TARGET_COLUMN)
    df_train[float_columns] = df_train[float_columns].astype("float32")
    df_test[float_columns] = df_test[float_columns].astype("float32")

    # Save preprocessed data
    df_train.to_csv(path_or_buf="Training_preprocessed.csv", index=False)
    df_test.to_csv(path_or_buf="Testing_preprocessed.csv", index=False)
