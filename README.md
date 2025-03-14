# Fine-tuned LLM Deployment

This repository contains a fine-tuned LLM (Large Language Model) for various NLP tasks such as grammatical error correction (GEC), paraphrasing, simplification, coherence enhancement, and more. The model has been fine-tuned using Unsloth for efficient training and deployed as a web API for inference.

## Features
- Fine-tuned transformer model using Unsloth for optimization.
- Supports multiple NLP tasks.
- Web API for easy integration.
- Hosted on Hugging Face and accessible via an endpoint.

## Table of Contents
- [Installation](#installation)
- [Fine-tuning Process](#fine-tuning-process)
- [Saving and Uploading the Model](#saving-and-uploading-the-model)
- [Deployment as Web API](#deployment-as-web-api)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
Clone this repository and install the dependencies:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
pip install -r requirements.txt
```

Ensure `requirements.txt` includes:
```txt
transformers
trl
unsloth
torch
fastapi
uvicorn
huggingface_hub
pandas
```

## Fine-tuning Process
The model was fine-tuned using Unsloth's `SFTTrainer`. The dataset was processed and mapped using:

```python
hf_train_dataset = hf_train_dataset.map(
    lambda x: {"formatted_text": f"{tokenizer.bos_token} {x['instruction']}\n\nProvide only the correct sentence after 'Output:', without any explanations.\n\nOutput: {x['output']} {eos_TOKEN}"}
)
```

Training was performed using:
```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=hf_train_dataset,
    dataset_text_field="formatted_text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args
)
```

## Saving and Uploading the Model
After fine-tuning, the model was saved and uploaded to Hugging Face:

```python
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
```
Then, upload to Hugging Face Hub:
```bash
huggingface-cli login
huggingface-cli upload ./fine_tuned_model --repo your-repo-name
```

## Deployment as Web API
To deploy the model as an API using FastAPI:

1. Create `app.py`:
```python
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()
model_name = "your-huggingface-repo"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_output(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/generate")
async def generate_text(data: dict):
    prompt = data.get("prompt")
    response = generate_output(prompt)
    return {"output": response}
```

2. Run the API:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Usage
To test the API, send a POST request:
```bash
curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Your input text here"}'
```

## Contributing
Feel free to submit issues or pull requests if you find improvements or bugs.

## License
This project is licensed under the MIT License.


