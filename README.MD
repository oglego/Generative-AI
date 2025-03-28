# Generative AI 

This repository houses various generative AI projects, with a primary focus on **text generation** using **Hugging Face Transformers** and **Flask** to create web-based applications. The first project in this repository is a **Flask app** that allows users to generate text using pre-trained models like GPT-2.

## Overview

The **HF-GPT2** app allows users to input a text prompt via a web interface or API and generates text based on that prompt using **Hugging Face's GPT-2 model**. The app is built using **Flask** for the backend and **Hugging Face's Transformers library** for text generation. 

## Features

- **Web Interface**: A simple interface for users to input a prompt and view the generated text.
- **REST API**: Allows users to send a `POST` request with a prompt and receive generated text.
- **Text Generation**: Uses a pre-trained model (GPT-2) to generate coherent and creative text based on the input prompt.

## Requirements

Before running the app, make sure you have the following installed:

- Python 3.x

## Running the app locally
Clone the repo

```bash
git clone https://github.com/oglego/generative-ai.git
cd hf-gpt2
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Run the app

```bash
python app.py
```

Open your browser and navigate to http://127.0.0.1:5000
