import requests
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from io import BytesIO
from pydantic import BaseModel, Field
from typing import List, Optional
import inferless
import torch
import json


@inferless.request
class RequestObjects(BaseModel):
    image_url: str = Field(default="https://raw.githubusercontent.com/HistAI/hibou/refs/heads/main/images/sample.png")

@inferless.response
class ResponseObjects(BaseModel):
    outputs: str = Field(default="")
    


class InferlessPythonModel:

    def initialize(self):
        self.processor = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)

    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        response = requests.get(inputs.image_url)
        image = Image.open(BytesIO(response.content))
        
        model_inputs = self.processor(images=image, return_tensors="pt")
        
        # Perform inference
        model_outputs = self.model(**model_inputs)
        
        # Extract last_hidden_state and convert to list
        if hasattr(model_outputs, "last_hidden_state") and isinstance(model_outputs.last_hidden_state, torch.Tensor):
            output_list = model_outputs.last_hidden_state.tolist()
        else:
            output_list = []  # Handle the case where last_hidden_state is missing
        
        return ResponseObjects(outputs=output_list)

    # perform any cleanup activity here
    def finalize(self,args):
        self.pipe = None
