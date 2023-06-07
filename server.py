"""Server that will listen for GET and POST requests from the client."""

import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from concrete.ml.deployment import FHEModelServer

# Initialize an instance of FastAPI
app = FastAPI()

current_dir = Path(__file__).parent

# Load the model
fhe_model = FHEModelServer(Path.joinpath(current_dir, "./client_folder"))

# Define the default route
@app.get("/")
def root():
    return {"message": "Welcome to Your disease prediction with fhe !"}


@app.post("/send_input")
def send_input(
    user_id: str = Form(),
    filter: str = Form(),
    files: List[UploadFile] = File(),
):
    """Send the inputs to the server."""
    # Retrieve the encrypted input image and the evaluation key paths
    encrypted_image_path = 0  # Tcurrent_dir("encrypted_image", user_id, filter)
    evaluation_key_path = current_dir / ".fhe_keys/{user_id}"

    # Write the files using the above paths
    with encrypted_image_path.open("wb") as encrypted_image, evaluation_key_path.open(
        "wb"
    ) as evaluation_key:
        encrypted_image.write(files[0].file.read())
        evaluation_key.write(files[1].file.read())


@app.post("/run_fhe")
def run_fhe(
    user_id: str = Form(),
    filter: str = Form(),
):
    """Execute the filter on the encrypted input image using FHE."""
    # Retrieve the encrypted input image and the evaluation key paths
    encrypted_image_path = get_server_file_path("encrypted_image", user_id, filter)
    evaluation_key_path = get_server_file_path("evaluation_key", user_id, filter)

    # Read the files using the above paths
    with encrypted_image_path.open("rb") as encrypted_image_file, evaluation_key_path.open(
        "rb"
    ) as evaluation_key_file:
        encrypted_image = encrypted_image_file.read()
        evaluation_key = evaluation_key_file.read()

    # Load the FHE server
    fhe_server = FHEServer(FILTERS_PATH / f"{filter}/deployment")

    # Run the FHE execution
    start = time.time()
    encrypted_output_image = fhe_server.run(encrypted_image, evaluation_key)
    fhe_execution_time = round(time.time() - start, 2)

    # Retrieve the encrypted output image path
    encrypted_output_path = get_server_file_path("encrypted_output", user_id, filter)

    # Write the file using the above path
    with encrypted_output_path.open("wb") as encrypted_output:
        encrypted_output.write(encrypted_output_image)

    return JSONResponse(content=fhe_execution_time)


@app.post("/get_output")
def get_output(
    user_id: str = Form(),
    filter: str = Form(),
):
    """Retrieve the encrypted output image."""
    # Retrieve the encrypted output image path
    encrypted_output_path = get_server_file_path("encrypted_output", user_id, filter)

    # Read the file using the above path
    with encrypted_output_path.open("rb") as encrypted_output_file:
        encrypted_output = encrypted_output_file.read()

    return Response(encrypted_output)
