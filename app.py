import subprocess
import time
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import requests
from symptoms_categories import SYMPTOMS_LIST
from utils import (
    CLIENT_DIR,
    CURRENT_DIR,
    DEPLOYMENT_DIR,
    INPUT_BROWSER_LIMIT,
    KEYS_DIR,
    SERVER_URL,
    clean_directory,
    get_disease_name,
    load_data,
    pretty_print,
)

from concrete.ml.deployment import FHEModelClient

subprocess.Popen(["uvicorn", "server:app"], cwd=CURRENT_DIR)
time.sleep(3)

# pylint: disable=c-extension-no-member,invalid-name


def is_none(obj) -> bool:
    """
    Check if the object is None.

    Args:
        obj (any): The input to be checked.

    Returns:
        bool: True if the object is None or empty, False otherwise.
    """
    return all((obj is None, (obj is not None and len(obj) < 1)))


# <!> This function has been paused due to UI issues.

# def fill_in_fn(default_disease: str, *checkbox_symptoms: Tuple[str]) -> Dict:
#     """
#     Fill in the gr.CheckBoxGroup list with predefined symptoms of a selected default disease.
#     Args:
#         default_disease (str): The default selected disease
#         *checkbox_symptoms (Tuple[str]): Existing checked symptoms
#     Returns:
#         dict: The updated gr.CheckBoxesGroup.
#     """
#
#     # Figure out the symptoms of the disease, selected by the user
#     df = pd.read_csv(TRAINING_FILENAME)
#     df_filtred = df[df[TARGET_COLUMNS[1]] == default_disease]
#     symptoms = pretty_print(df_filtred.columns[df_filtred.eq(1).any()].to_list())
#     # Check if there are existing symptoms, in the CheckbBxGroup list
#     if any(lst for lst in checkbox_symptoms if lst):
#         for sublist in checkbox_symptoms:
#             symptoms.extend(sublist)
#     return {box: symptoms for box in check_boxes}


def get_user_symptoms_from_checkboxgroup(checkbox_symptoms: List) -> np.array:
    """
    Convert the user symptoms into a binary vector representation.

    Args:
        checkbox_symptoms (List): A list of user symptoms.

    Returns:
        np.array: A binary vector representing the user's symptoms.

    Raises:
        KeyError: If a provided symptom is not recognized as a valid symptom.

    """
    symptoms_vector = {key: 0 for key in valid_symptoms}
    for pretty_symptom in checkbox_symptoms:
        original_symptom = "_".join((pretty_symptom.lower().split(" ")))
        if original_symptom not in symptoms_vector.keys():
            raise KeyError(
                f"The symptom '{original_symptom}' you provided is not recognized as a valid "
                f"symptom.\nHere is the list of valid symptoms: {symptoms_vector}"
            )
        symptoms_vector[original_symptom] = 1

    user_symptoms_vect = np.fromiter(symptoms_vector.values(), dtype=float)[np.newaxis, :]

    assert all(value == 0 or value == 1 for value in user_symptoms_vect.flatten())

    return user_symptoms_vect


def get_features_fn(*checked_symptoms: Tuple[str]) -> Dict:
    """
    Get vector features based on the selected symptoms.

    Args:
        checked_symptoms (Tuple[str]): User symptoms

    Returns:
        Dict: The encoded user vector symptoms.
    """
    if not any(lst for lst in checked_symptoms if lst):
        return {
            error_box1: gr.update(
                visible=True, value="Enter a default disease or select your own symptoms"
            ),
        }

    if len(pretty_print(checked_symptoms)) < 5:
        print("Provide at least 5 symptoms.")
        return {
            error_box1: gr.update(visible=True, value="Provide at least 5 symptoms"),
            user_vect_box1: get_user_symptoms_from_checkboxgroup([]),
        }


    return {
        error_box1: gr.update(visible=False),
        user_vect_box1: get_user_symptoms_from_checkboxgroup(pretty_print(checked_symptoms)),
    }


def key_gen_fn(user_symptoms: List[str]) -> Dict:
    """
    Generate keys for a given user.

    Args:
        user_symptoms (List[str]): The vector symptoms provided by the user.

    Returns:
        dict: A dictionary containing the generated keys and related information.

    """
    clean_directory()

    if is_none(user_symptoms):
        print("Error: Please submit your symptoms or select a default disease.")
        return {
            error_box2: gr.update(visible=True, value="Please submit your symptoms first!."),
        }

    # Generate a random user ID
    user_id = np.random.randint(0, 2**32)
    print(f"Your user ID is: {user_id}....")

    client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
    client.load()

    # Creates the private and evaluation keys on the client side
    client.generate_private_and_evaluation_keys()

    # Get the serialized evaluation keys
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert isinstance(serialized_evaluation_keys, bytes)

    # Save the evaluation key
    evaluation_key_path = KEYS_DIR / f"{user_id}/evaluation_key"
    with evaluation_key_path.open("wb") as f:
        f.write(serialized_evaluation_keys)

    serialized_evaluation_keys_shorten_hex = serialized_evaluation_keys.hex()[:INPUT_BROWSER_LIMIT]

    return {
        error_box2: gr.update(visible=False),
        key_box: serialized_evaluation_keys_shorten_hex,
        user_id_box: user_id,
        key_len_box: f"{len(serialized_evaluation_keys) / (10**6):.2f} MB",
    }


def encrypt_fn(user_symptoms: np.ndarray, user_id: str) -> None:
    """
    Encrypt the user symptoms vector in the `Client Side`.

    Args:
        user_symptoms (List[str]): The vector symptoms provided by the user
        user_id (user): The current user's ID
    """

    if is_none(user_id) or is_none(user_symptoms):
        print("Error in encryption step: Provide your symptoms and generate the evaluation keys.")
        return {
            error_box3: gr.update(
                visible=True, value="Please provide your symptoms and generate the evaluation keys."
            )
        }

    # Retrieve the client API
    client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
    client.load()

    user_symptoms = np.fromstring(user_symptoms[2:-2], dtype=int, sep=".").reshape(1, -1)
    quant_user_symptoms = client.model.quantize_input(user_symptoms)

    encrypted_quantized_user_symptoms = client.quantize_encrypt_serialize(user_symptoms)
    assert isinstance(encrypted_quantized_user_symptoms, bytes)
    encrypted_input_path = KEYS_DIR / f"{user_id}/encrypted_input"

    with encrypted_input_path.open("wb") as f:
        f.write(encrypted_quantized_user_symptoms)

    encrypted_quantized_user_symptoms_shorten_hex = encrypted_quantized_user_symptoms.hex()[
        :INPUT_BROWSER_LIMIT
    ]

    return {
        error_box3: gr.update(visible=False),
        user_vect_box2: user_symptoms,
        quant_vect_box: quant_user_symptoms,
        enc_vect_box: encrypted_quantized_user_symptoms_shorten_hex,
    }


def send_input_fn(user_id: str, user_symptoms: np.ndarray) -> Dict:
    """Send the encrypted data and the evaluation key to the server.

    Args:
        user_id (str): The current user's ID
        user_symptoms (np.ndarray): The user symptoms
    """

    if is_none(user_id) or is_none(user_symptoms):
        return {
            error_box4: gr.update(
                visible=True,
                value="Please ensure that the evaluation key has been generated "
                "and the symptoms have been submitted before sending the data to the server",
            )
        }

    evaluation_key_path = KEYS_DIR / f"{user_id}/evaluation_key"
    encrypted_input_path = KEYS_DIR / f"{user_id}/encrypted_input"

    if not evaluation_key_path.is_file():
        print(
            "Error Encountered While Sending Data to the Server: "
            f"The key has been generated correctly - {evaluation_key_path.is_file()=}"
        )

        return {error_box4: gr.update(visible=True, value="Please generate the private key first.")}

    if not encrypted_input_path.is_file():
        print(
            "Error Encountered While Sending Data to the Server: The data has not been encrypted "
            f"correctly on the client side - {encrypted_input_path.is_file()=}"
        )
        return {
            error_box4: gr.update(
                visible=True,
                value="Please encrypt the data with the private key first.",
            ),
        }

    # Define the data and files to post
    data = {
        "user_id": user_id,
        "input": user_symptoms,
    }

    files = [
        ("files", open(encrypted_input_path, "rb")),
        ("files", open(evaluation_key_path, "rb")),
    ]

    # Send the encrypted input and evaluation key to the server
    url = SERVER_URL + "send_input"
    with requests.post(
        url=url,
        data=data,
        files=files,
    ) as response:
        print(f"Sending Data: {response.ok=}")
    return {
        error_box4: gr.update(visible=False),
        srv_resp_send_data_box: "Data sent",
    }


def run_fhe_fn(user_id: str) -> Dict:
    """Send the encrypted input and the evaluation key to the server.

    Args:
        user_id (int): The current user's ID.
    """
    if is_none(user_id):
        return {
            error_box5: gr.update(
                visible=True,
                value="Please ensure that the evaluation key has been generated "
                "and the symptoms have been submitted before sending the data to the server",
            )
        }

    data = {
        "user_id": user_id,
    }

    url = SERVER_URL + "run_fhe"

    with requests.post(
        url=url,
        data=data,
    ) as response:
        if not response.ok:
            return {
                error_box5: gr.update(
                    visible=True,
                    value=(
                        "An error occurred on the Server Side. "
                        "Please check connectivity and data transmission."
                    ),
                ),
                fhe_execution_time_box: gr.update(visible=True),
            }
        else:
            print(f"response.ok: {response.ok}, {response.json()} - Computed")

    return {
        error_box5: gr.update(visible=False),
        fhe_execution_time_box: gr.update(value=f"{response.json()} seconds"),
    }


def get_output_fn(user_id: str, user_symptoms: np.ndarray) -> Dict:
    """Retreive the encrypted data from the server.

    Args:
        user_id (str): The current user's ID
        user_symptoms (np.ndarray): The user symptoms
    """

    if is_none(user_id) or is_none(user_symptoms):
        return {
            error_box6: gr.update(
                visible=True,
                value="Please ensure that the evaluation key has been generated "
                "and the symptoms have been submitted before sending the data to the server",
            )
        }

    data = {
        "user_id": user_id,
    }

    # Retrieve the encrypted output
    url = SERVER_URL + "get_output"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            print(f"Receive Data: {response.ok=}")

            encrypted_output = response.content

            # Save the encrypted output to bytes in a file as it is too large to pass through
            # regular Gradio buttons (see https://github.com/gradio-app/gradio/issues/1877)
            encrypted_output_path = CLIENT_DIR / f"{user_id}_encrypted_output"

            with encrypted_output_path.open("wb") as f:
                f.write(encrypted_output)
    return {error_box6: gr.update(visible=False), srv_resp_retrieve_data_box: "Data received"}


def decrypt_fn(user_id: str, user_symptoms: np.ndarray) -> Dict:
    """Dencrypt the data on the `Client Side`.

    Args:
        user_id (str): The current user's ID
        user_symptoms (np.ndarray): The user symptoms

    Returns:
        Decrypted output
    """

    if is_none(user_id) or is_none(user_symptoms):
        return {
            error_box7: gr.update(
                visible=True,
                value="Please ensure that the symptoms have been submitted and the evaluation "
                "key has been generated",
            )
        }

    # Get the encrypted output path
    encrypted_output_path = CLIENT_DIR / f"{user_id}_encrypted_output"

    if not encrypted_output_path.is_file():
        print("Error in decryption step: Please run the FHE execution, first.")
        return {
            error_box7: gr.update(
                visible=True,
                value="Please ensure that the symptoms have been submitted, the evaluation "
                "key has been generated and step 5 and 6 have been performed on the Server "
                "side before decrypting the prediction",
            )
        }

    # Load the encrypted output as bytes
    with encrypted_output_path.open("rb") as f:
        encrypted_output = f.read()

    # Retrieve the client API
    client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
    client.load()

    # Deserialize, decrypt and post-process the encrypted output
    output = client.deserialize_decrypt_dequantize(encrypted_output)


    print(output)

    return {
        error_box7: gr.update(visible=False),
        decrypt_target_box: get_disease_name(output.argmax()),
    }


def reset_fn():
    """Reset the space and clear all the box outputs."""

    clean_directory()

    return {
        # disease_box: None,
        user_id_box: None,
        user_vect_box1: None,
        user_vect_box2: None,
        quant_vect_box: None,
        enc_vect_box: None,
        key_box: None,
        key_len_box: None,
        fhe_execution_time_box: None,
        decrypt_target_box: None,
        error_box7: gr.update(visible=False),
        error_box1: gr.update(visible=False),
        error_box2: gr.update(visible=False),
        error_box3: gr.update(visible=False),
        error_box4: gr.update(visible=False),
        error_box5: gr.update(visible=False),
        error_box6: gr.update(visible=False),
        srv_resp_send_data_box: None,
        srv_resp_retrieve_data_box: None,
        **{box: None for box in check_boxes},
    }

def change_tab(next_tab):
    print(next_tab)
    return gr.Tabs.update(selected=next_tab)

CSS = """
/* #them {color: dark-yellow} */
/* #them {font-size: 25px}  */
/* #them {font-weight: bold}  */
.gradio-container {background-color: white}
/* .feedback {font-size: 3px !important} */
#svelte-s1r2yt {color: orange}
#svelte-s1r2yt {font-size: 25px}
#svelte-s1r2yt {font-weight: bold}
/* #them {text-align: center} */
"""

if __name__ == "__main__":

    print("Starting demo ...")

    clean_directory()

    (X_train, X_test), (y_train, y_test), valid_symptoms = load_data()

    with gr.Blocks(css=CSS) as demo:

        # Link + images
        gr.Markdown(
            """
            <p align="center">
                <img width=200 src="https://user-images.githubusercontent.com/5758427/197816413-d9cddad3-ba38-4793-847d-120975e1da11.png">
            </p>

            <h2 align="center">Health Prediction On Encrypted Data Using Fully Homomorphic Encryption.</h2>

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
            <img width="100%" height="30%" src="https://raw.githubusercontent.com/kcelia/Img/main/health_prediction_img.png">
            </p>
            """
        )

        with gr.Tabs(eelem_id ="svelte-s1r2yt", lem_classes="svelte-s1r2yt") as tabs:
            with gr.TabItem("1. Symptoms Selection", id=0):
                gr.Markdown("<span style='color:orange'>Client Side</span>")
                gr.Markdown("## Step 1: Provide your symptoms")
                gr.Markdown(
                    "You can provide your health condition either by checking "
                    "the symptoms available in the boxes or by selecting a known disease with "
                    "its predefined set of symptoms."
                )

                # Box symptoms
                check_boxes = []
                for i, category in enumerate(SYMPTOMS_LIST):
                    with gr.Accordion(
                        pretty_print(category.keys()),
                        open=False,
                        elem_classes="feedback",
                    ) as accordion:
                        check_box = gr.CheckboxGroup(
                            pretty_print(category.values()),
                            show_label=False,
                        )
                        check_boxes.append(check_box)

                error_box1 = gr.Textbox(label="Error", visible=False)

                # <!> This part has been paused due to UI issues.

                # Default disease, picked from the dataframe
                # disease_box = gr.Dropdown(list(sorted(set(df_test["prognosis"]))),
                # label="Disease:")
                # disease_box.change(
                #     fn=fill_in_fn,
                #     inputs=[disease_box, *check_boxes],
                #     outputs=[*check_boxes],
                # )

                # User symptom vector
                user_vect_box1 = gr.Textbox(label="User Symptoms Vector:", interactive=False)

                # Submit botton
                submit_button = gr.Button("Submit")

                submit_button.click(
                    fn=get_features_fn,
                    inputs=[*check_boxes],
                    outputs=[user_vect_box1, error_box1],
                )

                # Clear botton
                clear_button = gr.Button("Reset Space")

                next_tab = gr.Button('Next Step')
                next_tab.click(lambda _:gr.Tabs.update(selected=1), None, tabs)
                
            with gr.TabItem("2. Data Encryption", id=1):
                gr.Markdown("<span style='color:orange'>Client Side</span>")
                gr.Markdown("## Step 2: Generate the keys")

                gen_key_btn = gr.Button("Generate the keys")
                error_box2 = gr.Textbox(label="Error", visible=False)

                with gr.Row():
                    # User ID
                    with gr.Column(scale=1, min_width=600):
                        user_id_box = gr.Textbox(label="User ID:", interactive=False)
                    # Evaluation key size
                    with gr.Column(scale=1, min_width=600):
                        key_len_box = gr.Textbox(label="Evaluation Key Size:", interactive=False)

                # Evaluation key (truncated)
                with gr.Column(scale=2, min_width=600):
                    key_box = gr.Textbox(
                        label="Evaluation key (truncated):",
                        max_lines=3,
                        interactive=False,
                    )

                gen_key_btn.click(
                    key_gen_fn,
                    inputs=user_vect_box1,
                    outputs=[
                        key_box,
                        user_id_box,
                        key_len_box,
                        error_box2,
                    ],
                )

                gr.Markdown("## Step 3: Encrypt the symptoms")

                encrypt_btn = gr.Button("Encrypt the symptoms with the private key")
                error_box3 = gr.Textbox(label="Error", visible=False)


                with gr.Row():
                    with gr.Column(scale=1, min_width=600):
                        user_vect_box2 = gr.Textbox(
                            label="User Symptoms Vector:", interactive=False
                        )

                    with gr.Column(scale=1, min_width=600):
                        quant_vect_box = gr.Textbox(label="Quantized Vector:", interactive=False)

                    with gr.Column(scale=1, min_width=600):
                        enc_vect_box = gr.Textbox(
                            label="Encrypted Vector:", max_lines=3, interactive=False
                        )

                encrypt_btn.click(
                    encrypt_fn,
                    inputs=[user_vect_box1, user_id_box],
                    outputs=[
                        user_vect_box2,
                        quant_vect_box,
                        enc_vect_box,
                        error_box3,
                    ],
                )

                gr.Markdown(
                    "## Step 4: Send the encrypted data to the "
                    "<span style='color:orange'>Server Side</span>"
                )

                error_box4 = gr.Textbox(label="Error", visible=False)

                with gr.Row().style(equal_height=False):
                    with gr.Column(scale=4):
                        send_input_btn = gr.Button("Send the encrypted data")
                    with gr.Column(scale=1):
                        srv_resp_send_data_box = gr.Checkbox(
                            label="Data Sent", show_label=False, interactive=False
                        )

                send_input_btn.click(
                    send_input_fn,
                    inputs=[user_id_box, user_vect_box1],
                    outputs=[error_box4, srv_resp_send_data_box],
                )
                
                with gr.Row().style(equal_height=True):
                    with gr.Column(scale=1):
                        prev_tab = gr.Button('Previous Step')
                        prev_tab.click(lambda _:gr.Tabs.update(selected=0), None, tabs)

                    with gr.Column(scale=1):
                        next_tab = gr.Button('Next Step')
                        next_tab.click(lambda _:gr.Tabs.update(selected=2), None, tabs)
                


            with gr.TabItem("3. FHE execution", id=2):
                gr.Markdown("<span style='color:orange'>Server Side</span>")
                gr.Markdown("## Step 5: Run the FHE evaluation")

                run_fhe_btn = gr.Button("Run the FHE evaluation")
                error_box5 = gr.Textbox(label="Error", visible=False)
                fhe_execution_time_box = gr.Textbox(
                    label="Total FHE Execution Time:", interactive=False
                )

                run_fhe_btn.click(
                    run_fhe_fn,
                    inputs=[user_id_box],
                    outputs=[fhe_execution_time_box, error_box5],
                )

                with gr.Row().style(equal_height=True):
                    with gr.Column(scale=1):
                        prev_tab = gr.Button('Previous Step')
                        prev_tab.click(lambda _: gr.Tabs.update(selected=1), None, tabs)

                    with gr.Column(scale=1):
                        next_tab = gr.Button('Next Step')
                        next_tab.click(lambda _: gr.Tabs.update(selected=3), None, tabs)
                


            with gr.TabItem("4. Data Decryption", id=3):
                gr.Markdown("<span style='color:orange'>Client Side</span>")
                gr.Markdown(
                    "## Step 6: Get the data from the <span style='color:orange'>Server Side</span>"
                )

                error_box6 = gr.Textbox(label="Error", visible=False)

                with gr.Row().style(equal_height=True):
                    with gr.Column(scale=4):
                        get_output_btn = gr.Button("Get data")
                    with gr.Column(scale=1):
                        srv_resp_retrieve_data_box = gr.Checkbox(
                            label="Data Received", show_label=False, interactive=False
                        )

                get_output_btn.click(
                    get_output_fn,
                    inputs=[user_id_box, user_vect_box1],
                    outputs=[srv_resp_retrieve_data_box, error_box6],
                )

                gr.Markdown("## Step 7: Decrypt the output")

                decrypt_target_btn = gr.Button("Decrypt the output")
                error_box7 = gr.Textbox(label="Error", visible=False)
                decrypt_target_box = gr.Textbox(abel="Decrypted Output:", interactive=False)

                decrypt_target_btn.click(
                    decrypt_fn,
                    inputs=[user_id_box, user_vect_box1],
                    outputs=[decrypt_target_box, error_box7],
                )

                prev_tab = gr.Button('Previous Step')
                prev_tab.click(lambda _:gr.Tabs.update(selected=2), None, tabs)

        clear_button.click(
            reset_fn,
            outputs=[

                # disease_box,
                error_box1,
                error_box2,
                error_box3,
                error_box4,
                error_box5,
                error_box6,
                error_box7,
                user_id_box,
                key_len_box,
                key_box,
                quant_vect_box,
                enc_vect_box,
                srv_resp_send_data_box,
                srv_resp_retrieve_data_box,
                fhe_execution_time_box,
                decrypt_target_box,
                *check_boxes,
            ],
        )

        demo.launch()
