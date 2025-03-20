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




