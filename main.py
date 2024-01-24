import configparser
from fastapi import FastAPI
from pathlib import Path, PosixPath
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, pipeline
import torch
import uvicorn


def get_parent_dir():
    """
    Returns the parent directory of the current file.
    """
    return Path(__file__).parent

config = configparser.ConfigParser()
config.read(get_parent_dir() / "config.ini")
model_dir = config["DEFAULT"].get("ModelDir")
model_name = config["DEFAULT"].get("ModelName")

app = FastAPI()
models_path = get_parent_dir() / model_dir / model_name

# Path to your fine-tuned model and tokenizer

if len(model_dir) == 0:
    raise ValueError("Please specify a model dir in config.ini")

tokenizer = AutoTokenizer.from_pretrained(models_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier = pipeline('sentiment-analysis',
                      model=models_path,
                      device=device,
                      truncation=True
                     )

@app.post("/predict")
async def detect(text: str):
    """Label a single text"""
    predictions = classifier.predict(text)
    return {"prediction": predictions[0],
            "model": model_name}


@app.post("/predict_batch")
async def detect_batch(texts: list[str]):
    """Label a batch of texts"""
    predictions = classifier.predict(texts)
    return {"prediction": predictions,
            "model": model_name}


if __name__ == "__main__":
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8036,
                reload=True)
