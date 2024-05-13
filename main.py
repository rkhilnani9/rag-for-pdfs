"""
main app file
"""
import os
import shutil

from pathlib import Path
from ai_compendium.io_models import InferenceData, InferenceOutput, CreateEmbeddingData
from fastapi import FastAPI, HTTPException, File, UploadFile
from ai_compendium.api.predict_response import get_query_response
from ai_compendium.api.create_embeddings import generate_embeddings
from ai_compendium.business_excepts import custom_except

app = FastAPI()


@app.get("/health-check")
async def health_check():
    try:
        result = {"status_code": 200, "message": "Server Up"}
    except Exception as e:
        result = custom_except(message=str(e), status_code=500)
    return result


@app.post("/upload-file")
async def upload(input_file: UploadFile = File(...)):
    """upload_file : Reads a pdf file and saves it to disk

    Returns:
        _json_: status (SUCCESS or FAILURE) along with filename of uploaded pdf
    """

    # try:
    os.makedirs("pdfs/", exist_ok=True)
    file_name = input_file.filename
    destination = Path(f"pdfs/{file_name}")
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(input_file.file, buffer)
    finally:
        input_file.file.close()
    return {"status": "SUCCESS", "output": destination}
    # except Exception as ex:
    #     return {"status": "FAILURE", "output": ex}


@app.post("/get-query-response")
async def predict_response(request_data: InferenceData):
    try:
        # Extract the parameters from the request body
        # todo: add hotel_id, guest_id to request body
        query_text = request_data.query_text

        # Call the perform_task function with the extracted parameters
        output_text = get_query_response(query_text, doc_id)

        # Return the result as a JSON response
        response = InferenceOutput(status_code=200, data=output_text, message="Success")
    except ValueError as ve:
        # Catch specific exceptions (e.g., ValueError) and handle them with appropriate status codes and error messages
        response = InferenceOutput(status_code=400, data="", message=str(ve))
    except Exception as e:
        # Catch any other unexpected exceptions and return a generic error message with status code 500
        response = InferenceOutput(status_code=500, data="", message=str(e))
    return response


@app.post("/create-embeddings")
async def get_embedding_table(request_data: CreateEmbeddingData):
    try:
        # Extract the parameters from the request body
        source_file_path = request_data.source_file_path
        doc_id = request_data.doc_id

        # Call the perform_task function with the extracted parameters
        # saved_embeddings_path = create_embeddings_from_text(source_file_path)
        generate_embeddings(source_file_path, doc_id)

        # Return the result as a JSON response
        response = InferenceOutput(
            status_code=200,
            data="Embedding table created successfully",
            message="Success",
        )
    except ValueError as ve:
        # Catch specific exceptions (e.g., ValueError) and handle them with appropriate status codes and error messages
        response = InferenceOutput(status_code=400, data="", message=str(ve))
    except Exception as e:
        # Catch any other unexpected exceptions and return a generic error message with status code 500
        response = InferenceOutput(status_code=500, data="", message=str(e))
    return response
