import pickle as pkl
import shutil
from pathlib import Path
from time import time
from typing import List, Tuple, Union

import gradio as gr
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.model_selection import train_test_split

from concrete.ml.common.serialization.loaders import load, loads
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBoostClassifier

path_to_model = Path("./client_folder").resolve()

import subprocess

from preprocessing import (  # pylint: disable=wrong-import-position, no-name-in-module
    map_prediction,
    pretty_print,
)
from symptoms_categories import SYMPTOMS_LIST

ENCRYPTED_DATA_BROWSER_LIMIT = 500
# This repository's directory
REPO_DIR = Path(__file__).parent

print(f"{REPO_DIR=}")
# subprocess.Popen(["uvicorn", "server:app"], cwd=REPO_DIR)
# time.sleep(3)


def load_data():
    # Load data
    df_train = pd.read_csv("./data/Training_preprocessed.csv")
    df_test = pd.read_csv("./data/Testing_preprocessed.csv")

    # Separate the traget from the training set
    # df['prognosis] contains the name of the disease
    # df['y] contains the numeric label of the disease

    y_train = df_train["y"]
    X_train = df_train.drop(columns=["y", "prognosis"], axis=1, errors="ignore")

    y_test = df_train["y"]
    X_test = df_test.drop(columns=["y", "prognosis"], axis=1, errors="ignore")

    return (df_train, X_train, X_test), (df_test, y_train, y_test)


def load_model(X_train, y_train):
    concrete_args = {"max_depth": 1, "n_bits": 3, "n_estimators": 3, "n_jobs": -1}
    classifier = ConcreteXGBoostClassifier(**concrete_args)
    classifier.fit(X_train, y_train)
    circuit = classifier.compile(X_train)

    return classifier, circuit


def key_gen():

    # Key serialization
    user_id = np.random.randint(0, 2**32)

    client = FHEModelClient(path_dir=path_to_model, key_dir=f".fhe_keys/{user_id}")
    client.load()

    # The client first need to create the private and evaluation keys.

    client.generate_private_and_evaluation_keys()

    # Get the serialized evaluation keys
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert isinstance(serialized_evaluation_keys, bytes)

    np.save(f".fhe_keys/{user_id}/eval_key.npy", serialized_evaluation_keys)

    serialized_evaluation_keys_shorten = list(serialized_evaluation_keys)[:200]
    serialized_evaluation_keys_shorten_hex = "".join(
        f"{i:02x}" for i in serialized_evaluation_keys_shorten
    )
    # Evaluation keys can be quite large files but only have to be shared once with the server.

    # Check the size of the evaluation keys (in MB)
    return [
        serialized_evaluation_keys_shorten_hex,
        user_id,
        f"{len(serialized_evaluation_keys) / (10**6):.2f} MB",
    ]


def encode_quantize_encrypt(user_symptoms, user_id):
    # check if the key has been generated
    client = FHEModelClient(path_dir=path_to_model, key_dir=f".fhe_keys/{user_id}")
    client.load()

    user_symptoms = np.fromstring(user_symptoms[2:-2], dtype=int, sep=".").reshape(1, -1)

    quant_user_symptoms = client.model.quantize_input(user_symptoms)
    encrypted_quantized_user_symptoms = client.quantize_encrypt_serialize(user_symptoms)

    # print(client.model.predict(vect_x, fhe="simulate"), client.model.predict(vect_x, fhe="execute"))
    # pred_s = client.model.fhe_circuit.simulate(quant_vect)
    # pred_fhe = client.model.fhe_circuit.encrypt_run_decrypt(quant_vect) #
    # non alpha -> \X1124, base64 ou en exa

    # Compute size

    np.save(f".fhe_keys/{user_id}/encrypted_quant_vect.npy", encrypted_quantized_user_symptoms)

    encrypted_quantized_encoding_shorten = list(encrypted_quantized_user_symptoms)[:200]
    encrypted_quantized_encoding_shorten_hex = "".join(
        f"{i:02x}" for i in encrypted_quantized_encoding_shorten
    )

    return user_symptoms, quant_user_symptoms, encrypted_quantized_encoding_shorten_hex


def decrypt_prediction(encrypted_quantized_vect, user_id):
    fhe_api = FHEModelClient(path_dir=path_to_model, key_dir=f".fhe_keys/{user_id}")
    fhe_api.load()
    fhe_api.generate_private_and_evaluation_keys(force=False)
    predictions = fhe_api.deserialize_decrypt_dequantize(encrypted_quantized_vect)
    return predictions


def get_user_vect_symptoms_from_checkboxgroup(*user_symptoms) -> np.array:
    symptoms_vector = {key: 0 for key in valid_columns}

    for symptom_box in user_symptoms:
        for pretty_symptom in symptom_box:
            symptom = "_".join((pretty_symptom.lower().split(" ")))
            if symptom not in symptoms_vector.keys():
                raise KeyError(
                    f"The symptom '{symptom}' you provided is not recognized as a valid "
                    f"symptom.\nHere is the list of valid symptoms: {symptoms_vector}"
                )
            symptoms_vector[symptom] = 1.0

    user_symptoms_vect = np.fromiter(symptoms_vector.values(), dtype=float)[np.newaxis, :]

    assert all(value == 0 or value == 1 for value in user_symptoms_vect.flatten())

    return user_symptoms_vect


def get_user_vect_symptoms_from_default_disease(disease):

    user_symptom_vector = df_test[df_test["prognosis"] == disease].iloc[0].values

    user_symptoms_vect = np.fromiter(user_symptom_vector[:-2], dtype=float)[np.newaxis, :]

    assert all(value == 0 or value == 1 for value in user_symptoms_vect.flatten())

    return user_symptoms_vect


def get_user_symptoms_from_default_disease(disease):
    df_filtred = df_test[df_test["prognosis"] == disease]
    columns_with_1 = df_filtred.columns[df_filtred.eq(1).any()].to_list()
    return pretty_print(columns_with_1)


def get_user_symptoms_vector(selected_default_disease, *selected_symptoms):

    if any(lst for lst in selected_symptoms if lst) and (
        selected_default_disease is not None and len(selected_default_disease) > 0
    ):
        # If the user has already selected a disease and added more symptoms, raise an error
        if set(pretty_print(selected_symptoms)) - set(
            get_user_symptoms_from_default_disease(selected_default_disease)
        ):
            return {
                user_vector_textbox: gr.update(value="An error occurs"),
                error_box: gr.update(
                    visible=True, value="Enter a default disease or select your own symptoms"
                ),
            }
    # If the user has not selected a default disease or symptoms, an error is raised.
    if not any(lst for lst in selected_symptoms if lst) and (
        selected_default_disease is None
        or (selected_default_disease is not None and len(selected_default_disease) < 1)
    ):
        return {
            user_vector_textbox: gr.update(value="An error occurs"),
            error_box: gr.update(
                visible=True, value="Enter a default disease or select your own symptoms"
            ),
        }
    # Case 1: The user has checked his own symptoms
    if any(lst for lst in selected_symptoms if lst):
        return {
            user_vector_textbox: get_user_vect_symptoms_from_checkboxgroup(*selected_symptoms),
        }

    # Case 2: The user has selected a default disease
    if selected_default_disease is not None and len(selected_default_disease) > 0:
        return {
            user_vector_textbox: get_user_vect_symptoms_from_default_disease(
                selected_default_disease
            ),
            error_box: gr.update(visible=False),
            **{
                box: get_user_symptoms_from_default_disease(selected_default_disease)
                for box in check_boxes
            },
        }


def clear_all_buttons():
    return {
        user_id_textbox: None,
        eval_key_textbox: None,
        eval_key_len_textbox: None,
        user_vector_textbox: None,
        box_default: None,
        error_box: gr.update(visible=False),
        **{box: None for box in check_boxes},
    }


if __name__ == "__main__":
    print("Starting demo ...")

    (df_train, X_train, X_test), (df_test, y_train, y_test) = load_data()

    valid_columns = X_train.columns.to_list()

    with gr.Blocks() as demo:

        # Link + images
        gr.Markdown(
            """
    <p align="center">
        <img width=200 src="https://user-images.githubusercontent.com/5758427/197816413-d9cddad3-ba38-4793-847d-120975e1da11.png">
    </p>

    <h2 align="center">Health Prediction On Encrypted Data Using Homomorphic Encryption.</h2>

    <p align="center">
        <a href="https://github.com/zama-ai/concrete-ml"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197972109-faaaff3e-10e2-4ab6-80f5-7531f7cfb08f.png">Concrete-ML</a>
        —
        <a href="https://docs.zama.ai/concrete-ml"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197976802-fddd34c5-f59a-48d0-9bff-7ad1b00cb1fb.png">Documentation</a>
        —
        <a href="https://zama.ai/community"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197977153-8c9c01a7-451a-4993-8e10-5a6ed5343d02.png">Community</a>
        —
        <a href="https://twitter.com/zama_fhe"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197975044-bab9d199-e120-433b-b3be-abd73b211a54.png">@zama_fhe</a>
    </p>

    <p align="center">
    <img src="https://raw.githubusercontent.com/kcelia/Img/main/demo-img2.png" width="60%" height="60%">
    </p>
    """
        )

        # Gentle introduction
        gr.Markdown("## Introduction")
        gr.Markdown("""Blablabla""")

        # User symptoms
        gr.Markdown("# Step 1: Provide your symptoms")
        gr.Markdown("Client side")

        # Default disease, picked from the dataframe
        with gr.Row():
            default_diseases = list(set(df_test["prognosis"]))
            box_default = gr.Dropdown(default_diseases, label="Disease")

        # Box symptoms
        check_boxes = []
        for i, category in enumerate(SYMPTOMS_LIST):
            check_box = gr.CheckboxGroup(
                pretty_print(category.values()),
                label=pretty_print(category.keys()),
                info=f"Symptoms related to `{pretty_print(category.values())}`",
                max_batch_size=45,
            )
            check_boxes.append(check_box)

        # User symptom vector
        with gr.Row():
            user_vector_textbox = gr.Textbox(
                label="User symptoms (vector)",
                interactive=False,
                max_lines=100,
            )
        error_box = gr.Textbox(label="Error", visible=False)

        with gr.Row():
            # Submit botton
            with gr.Column():
                submit_button = gr.Button("Submit")
            # Clear botton
            with gr.Column():
                clear_button = gr.Button("Clear", style="background-color: yellow;")

        # Click submit botton

        submit_button.click(
            fn=get_user_symptoms_vector,
            inputs=[box_default, *check_boxes],
            outputs=[user_vector_textbox, error_box, *check_boxes],
        )
        # Load the model
        concrete_classifier = load(
            open("ConcreteXGBoostClassifier.pkl", "r", encoding="utf-8")
        )

        gr.Markdown("# Step 2: Generate the keys")
        gr.Markdown("Client side")

        gen_key = gr.Button("Generate the keys and send public part to server")

        with gr.Row():
            # User ID
            with gr.Column(scale=1, min_width=600):
                user_id_textbox = gr.Textbox(
                    label="User ID:",
                    max_lines=4,
                    interactive=False,
                )
            # Evaluation key size
            with gr.Column(scale=1, min_width=600):
                eval_key_len_textbox = gr.Textbox(
                    label="Evaluation key size:", max_lines=4, interactive=False
                )

        with gr.Row():
            # Evaluation key (truncated)
            with gr.Column(scale=2, min_width=600):
                eval_key_textbox = gr.Textbox(
                    label="Evaluation key (truncated):",
                    max_lines=4,
                    interactive=False,
                )

        gen_key.click(key_gen, outputs=[eval_key_textbox, user_id_textbox, eval_key_len_textbox])

        clear_button.click(
            clear_all_buttons,
            outputs=[
                user_id_textbox,
                user_vector_textbox,
                eval_key_textbox,
                eval_key_len_textbox,
                box_default,
                error_box,
                *check_boxes,
            ],
        )

        gr.Markdown("# Step 3: Encode the message with the private key")
        gr.Markdown("Client side")

        encode_msg = gr.Button("Generate the keys and send public part to server")

        with gr.Row():

            with gr.Column(scale=1, min_width=600):
                vect_textbox = gr.Textbox(
                    label="Vector:",
                    max_lines=4,
                    interactive=False,
                )

            with gr.Column(scale=1, min_width=600):
                quant_vect_textbox = gr.Textbox(
                    label="Quant vector:", max_lines=4, interactive=False
                )

            with gr.Column(scale=1, min_width=600):
                encrypted_vect_textbox = gr.Textbox(
                    label="Encrypted vector:", max_lines=4, interactive=False
                )

        encode_msg.click(
            encode_quantize_encrypt,
            inputs=[user_vector_textbox, user_id_textbox],
            outputs=[vect_textbox, quant_vect_textbox, encrypted_vect_textbox],
        )

        gr.Markdown("# Step 4: Run the FHE evaluation")
        gr.Markdown("Server side")

        run_fhe = gr.Button("Run the FHE evaluation")

        gr.Markdown("# Step 5: Decrypt the sentiment")
        gr.Markdown("Server side")

        decrypt_target_botton = gr.Button("Decrypt the sentiment")
        decrypt_target_textbox = gr.Textbox(
            label="Encrypted vector:", max_lines=4, interactive=False
        )

        decrypt_target_botton.click(
            decrypt_prediction,
            inputs=[encrypted_vect_textbox, user_id_textbox],
            outputs=[decrypt_target_textbox],
        )

    demo.launch()
