#!/usr/bin/env python
import os
import time
import uuid
import requests
from typing import Literal, Optional, Dict, Any

import runpod
from pydantic import BaseModel, Field

# We keep the original engine and server startup logic
from engine import SGlangEngine

# --- 1. DEFINE YOUR CUSTOM API SCHEMAS ---

# This is the input your API will accept.
# It's simplified to focus on the core "generate" functionality.
class APIInput(BaseModel):
    prompt: str = Field(
        ...,
        description="The main text prompt for the language model.",
        example="Explain the theory of relativity in simple terms."
    )
    sampling_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of sampling parameters like max_new_tokens, temperature, etc."
    )

# This is the exact JSON structure your API will return.
class APIOutput(BaseModel):
    request_id: str = Field(..., description="Unique ID for the request.")
    model_name: str = Field(..., description="The name of the model that processed the request.")
    execution_time_ms: int = Field(..., description="Total processing time in milliseconds.")
    status: Literal["success", "error"] = Field(..., description="The status of the request.")
    text_output: str = Field(..., description="The complete generated text from the model.")
    error_message: Optional[str] = Field(None, description="Details if the status is 'error'.")


# --- 2. INITIALIZE THE ENGINE AND SERVER ---
# This part is kept from the original file. It starts the background SGLang server.
engine = SGlangEngine()
engine.start_server()
engine.wait_for_server()


# --- 3. THE NEW, CUSTOMIZED HANDLER FUNCTION ---
def handler(job: dict):
    """
    This is the main handler function. It's now synchronous and returns a single
    JSON object with a custom structure.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    model_name = os.environ.get("MODEL_NAME", "unknown")

    # --- Input Validation ---
    try:
        validated_input = APIInput.parse_obj(job['input'])
    except Exception as e:
        end_time = time.time()
        error_payload = APIOutput(
            request_id=request_id,
            model_name=model_name,
            execution_time_ms=int((end_time - start_time) * 1000),
            status="error",
            text_output="",
            error_message=f"Input validation failed: {e}"
        )
        return error_payload.model_dump()

    # --- Prepare Request for Local SGLang Server ---
    # We will use the SGLang server's native "/generate" endpoint.
    generate_url = f"{engine.base_url}/generate"
    headers = {"Content-Type": "application/json"}

    # The payload for the /generate endpoint includes the prompt and sampling params.
    # We explicitly disable streaming to get a single, complete response.
    payload = {
        "text": validated_input.prompt,
        "sampling_params": {
            **validated_input.sampling_params,
            "stream": False  # IMPORTANT: Ensure non-streaming response
        }
    }

    # --- Run Inference by Calling the Local Server ---
    try:
        response = requests.post(generate_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # The SGLang /generate endpoint returns a JSON with a "text" key.
        sglang_response_data = response.json()
        generated_text = sglang_response_data.get("text", "")

    except requests.exceptions.RequestException as e:
        end_time = time.time()
        error_payload = APIOutput(
            request_id=request_id,
            model_name=model_name,
            execution_time_ms=int((end_time - start_time) * 1000),
            status="error",
            text_output="",
            error_message=f"Request to SGLang server failed: {e}"
        )
        return error_payload.model_dump()

    end_time = time.time()
    execution_time_ms = int((end_time - start_time) * 1000)

    # --- Construct and Return the Custom Output ---
    success_payload = APIOutput(
        request_id=request_id,
        model_name=model_name,
        execution_time_ms=execution_time_ms,
        status="success",
        text_output=generated_text
    )

    return success_payload.model_dump()


# --- 4. START THE RUNPOD SERVERLESS WORKER ---
# We configure Runpod to use our new synchronous handler.
runpod.serverless.start({
    "handler": handler
})
