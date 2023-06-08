import os
import shutil
import subprocess
from pathlib import Path
from time import time
from typing import List, Tuple, Union

import gradio as gr
import numpy as np
import pandas as pd
from preprocessing import pretty_print
from symptoms_categories import SYMPTOMS_LIST

from concrete.ml.common.serialization.loaders import load
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBoostClassifier

INPUT_BROWSER_LIMIT = 635

# This repository's main necessary folders
REPO_DIR = Path(__file__).parent
MODEL_PATH = REPO_DIR / "client_folder"
KEYS_PATH = REPO_DIR / ".fhe_keys"
CLIENT_PATH = MODEL_PATH / "client.zip"
SERVER_PATH = MODEL_PATH / "server.zip"

# subprocess.Popen(["uvicorn", "server:app"], cwd=REPO_DIR)
# time.sleep(3)


def clean_directory():
    target_dir = ".fhe_keys"
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
        print("The .fhe_keys directory and its contents have been successfully removed.")
    else:
        print("The .keys directory does not exist.")


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


def get_user_vect_symptoms_from_checkboxgroup(*user_symptoms) -> np.array:
    symptoms_vector = {key: 0 for key in VALID_COLUMNS}

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


def get_user_vector_from_default_disease(disease):

    user_symptom_vector = df_test[df_test["prognosis"] == disease].iloc[0].values

    user_symptoms_vect = np.fromiter(user_symptom_vector[:-2], dtype=float)[np.newaxis, :]

    assert all(value == 0 or value == 1 for value in user_symptoms_vect.flatten())

    return user_symptoms_vect


def get_user_symptoms_from_default_disease(disease):
    df_filtred = df_test[df_test["prognosis"] == disease]
    columns_with_1 = df_filtred.columns[df_filtred.eq(1).any()].to_list()
    return pretty_print(columns_with_1)


def get_user_symptoms_vector_fn(selected_default_disease, *selected_symptoms):

    # Display an error box, if:
    # 1. The user has already selected a default disease and added more symptoms, or
    # 2. The the user has not selected a default disease or symptoms
    if (
        any(lst for lst in selected_symptoms if lst)
        and (selected_default_disease is not None and len(selected_default_disease) > 0)
        and set(pretty_print(selected_symptoms))
        - set(get_user_symptoms_from_default_disease(selected_default_disease))
    ) or (
        not any(lst for lst in selected_symptoms if lst)
        and (
            selected_default_disease is None
            or (selected_default_disease is not None and len(selected_default_disease) < 1)
        )
    ):
        return {
            error_box_1: gr.update(
                visible=True, value="Enter a default disease or select your own symptoms"
            ),
        }
    # Case 1: The user has checked his own symptoms
    if any(lst for lst in selected_symptoms if lst):
        return {
            error_box_1: gr.update(visible=False),
            user_vector_textbox: get_user_vect_symptoms_from_checkboxgroup(*selected_symptoms),
        }

    # Case 2: The user has selected a default disease
    if selected_default_disease is not None and len(selected_default_disease) > 0:
        return {
            user_vector_textbox: get_user_vector_from_default_disease(selected_default_disease),
            error_box_1: gr.update(visible=False),
            **{
                box: get_user_symptoms_from_default_disease(selected_default_disease)
                for box in check_boxes
            },
        }


def key_gen_fn(user_symptoms):

    print("Cleaning directory ...")
    clean_directory()

    if user_symptoms is None or (user_symptoms is not None and len(user_symptoms) < 1):
        print("Please submit your symptoms first")
        return {
            error_box_2: gr.update(visible=True, value="Please submit your symptoms first"),
        }

    # Key serialization
    user_id = np.random.randint(0, 2**32)

    client = FHEModelClient(path_dir=MODEL_PATH, key_dir=KEYS_PATH / f"{user_id}")
    client.load()

    # The client first need to create the private and evaluation keys.

    client.generate_private_and_evaluation_keys()

    # Get the serialized evaluation keys
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert isinstance(serialized_evaluation_keys, bytes)

    # np.save(f".fhe_keys/{user_id}/eval_key.npy", serialized_evaluation_keys)
    evaluation_key_path = KEYS_PATH / f"{user_id}/evaluation_key"
    with evaluation_key_path.open("wb") as evaluation_key_file:
        evaluation_key_file.write(serialized_evaluation_keys)

    serialized_evaluation_keys_shorten_hex = serialized_evaluation_keys.hex()[:INPUT_BROWSER_LIMIT]

    return {
        error_box_2: gr.update(visible=False),
        eval_key_textbox: serialized_evaluation_keys_shorten_hex,
        user_id_textbox: user_id,
        eval_key_len_textbox: f"{len(serialized_evaluation_keys) / (10**6):.2f} MB",
    }


def encrypt_fn(user_symptoms, user_id):

    if not user_symptoms or not user_symptoms:
        return {
            error_box_3: gr.update(
                visible=True, value="Please ensure that the evaluation key has been generated!"
            )
        }

    # Retrieve the client API

    client = FHEModelClient(path_dir=MODEL_PATH, key_dir=KEYS_PATH / f"{user_id}")
    client.load()

    user_symptoms = np.fromstring(user_symptoms[2:-2], dtype=int, sep=".").reshape(1, -1)

    quant_user_symptoms = client.model.quantize_input(user_symptoms)
    encrypted_quantized_user_symptoms = client.quantize_encrypt_serialize(user_symptoms)

    encrypted_input_path = KEYS_PATH / f"{user_id}/encrypted_symptoms"

    with encrypted_input_path.open("wb") as f:
        f.write(encrypted_quantized_user_symptoms)

    # print(client.model.predict(vect_x, fhe="simulate"), client.model.predict(vect_x, fhe="execute"))
    # pred_s = client.model.fhe_circuit.simulate(quant_vect)
    # pred_fhe = client.model.fhe_circuit.encrypt_run_decrypt(quant_vect) #
    # non alpha -> \X1124, base64 ou en exa

    # Compute size

    # np.save(f".fhe_keys/{user_id}/encrypted_quant_vect.npy", encrypted_quantized_user_symptoms)

    encrypted_quantized_user_symptoms_shorten_hex = encrypted_quantized_user_symptoms.hex()[
        :INPUT_BROWSER_LIMIT
    ]

    return {
        error_box_3: gr.update(visible=False),
        vect_textbox: user_symptoms,
        quant_vect_textbox: quant_user_symptoms,
        encrypted_vect_textbox: encrypted_quantized_user_symptoms_shorten_hex,
    }


# def send_input(user_id, user_symptoms):
#     """Send the encrypted input image as well as the evaluation key to the server.

#     Args:
#         user_id (int): The current user's ID.
#         filter_name (str): The current filter to consider.
#     """
#     # Get the evaluation key path


#     evaluation_key_path = get_client_file_path("evaluation_key", user_id, filter_name)

#     if user_id == "" or not evaluation_key_path.is_file():
#         raise gr.Error("Please generate the private key first.")

#     encrypted_input_path = get_client_file_path("encrypted_image", user_id, filter_name)
#     encrypted_symptoms_path = KEYS_PATH / f"{user_id}" / "encrypted_symtoms"

#     if not encrypted_input_path.is_file():
#         raise gr.Error("Please generate the private key and then encrypt an image first.")

#     # Define the data and files to post
#     data = {
#         "user_id": user_id,
#         "filter": filter_name,
#     }

#     files = [
#         ("files", open(encrypted_input_path, "rb")),
#         ("files", open(evaluation_key_path, "rb")),
#     ]

#     # Send the encrypted input image and evaluation key to the server
#     url = SERVER_URL + "send_input"
#     with requests.post(
#         url=url,
#         data=data,
#         files=files,
#     ) as response:
#         return response.ok


# def decrypt_prediction(encrypted_quantized_vect, user_id):
#     fhe_api = FHEModelClient(path_dir=REPO_DIR, key_dir=f".fhe_keys/{user_id}")
#     fhe_api.load()
#     fhe_api.generate_private_and_evaluation_keys(force=False)
#     predictions = fhe_api.deserialize_decrypt_dequantize(encrypted_quantized_vect)
#     return predictions




def clear_all_btn():
    return {
        box_default: None,
        user_id_textbox: None,
        eval_key_textbox: None,
        quant_vect_textbox: None,
        user_vector_textbox: None,
        eval_key_len_textbox: None,
        encrypted_vect_textbox: None,
        error_box_1: gr.update(visible=False),
        error_box_2: gr.update(visible=False),
        error_box_3: gr.update(visible=False),
        **{box: None for box in check_boxes},
    }


if __name__ == "__main__":
    print("Starting demo ...")
    

    (df_train, X_train, X_test), (df_test, y_train, y_test) = load_data()

    VALID_COLUMNS = X_train.columns.to_list()

    # Load the model
    with open("ConcreteXGBoostClassifier.pkl", "r", encoding="utf-8") as file:
        concrete_classifier = load(file)

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

        error_box_1 = gr.Textbox(label="Error", visible=False)

        # User symptom vector
        with gr.Row():
            user_vector_textbox = gr.Textbox(
                label="User symptoms (vector)",
                interactive=False,
                max_lines=100,
            )

        with gr.Row():
            # Submit botton
            with gr.Column():
                submit_button = gr.Button("Submit")
            # Clear botton
            with gr.Column():
                clear_button = gr.Button("Clear")

        # Click submit botton

        submit_button.click(
            fn=get_user_symptoms_vector_fn,
            inputs=[box_default, *check_boxes],
            outputs=[user_vector_textbox, error_box_1, *check_boxes],
        )

        gr.Markdown("# Step 2: Generate the keys")
        gr.Markdown("Client side")

        gen_key_btn = gr.Button("Generate the keys and send public part to server")

        error_box_2 = gr.Textbox(label="Error", visible=False)

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

        gen_key_btn.click(
            key_gen_fn,
            inputs=user_vector_textbox,
            outputs=[eval_key_textbox, user_id_textbox, eval_key_len_textbox, error_box_2],
        )

        gr.Markdown("# Step 3: Encode the message with the private key")
        gr.Markdown("Client side")

        encrypt_btn = gr.Button("Encode the message with the private key and send it to the server")

        error_box_3 = gr.Textbox(label="Error", visible=False)

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

        encrypt_btn.click(
            encrypt_fn,
            inputs=[user_vector_textbox, user_id_textbox],
            outputs=[vect_textbox, quant_vect_textbox, encrypted_vect_textbox, error_box_3],
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

        # decrypt_target_botton.click(
        #     decrypt_prediction,
        #     inputs=[encrypted_vect_textbox, user_id_textbox],
        #     outputs=[decrypt_target_textbox],
        # )

        clear_button.click(
            clear_all_btn,
            outputs=[
                box_default,
                error_box_1,
                error_box_2,
                error_box_3,
                user_id_textbox,
                eval_key_textbox,
                quant_vect_textbox,
                user_vector_textbox,
                eval_key_len_textbox,
                encrypted_vect_textbox,
                *check_boxes,
            ],
        )

    demo.launch()
