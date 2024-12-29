# Deep Neural Network for AI detection in texts
This project leverages deep neural networks to analyze text made by LLM . Built using powerful machine learning frameworks such as sklearn and PyTorch and OpenAI API, this tool aims to detect AI ingerency in everyday life.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
Exploring the capability of artificial intelligence to distinguish between human-written and AI-generated texts is a compelling and innovative field of study. This project employs advanced deep learning techniques to classify and identify text origins, aiming to accurately differentiate between content created by humans and that produced by AI models. By leveraging sentiment analysis and embedding models, the project aspires to enhance our understanding of textual characteristics and the nuances that separate human and AI-generated content.

## Features
- Detection and differentiation of AI-generated texts versus human-written content

- Built OpenAI Embedding for advanced natural language processing

- Leveraging PyTorch for deep learning and neural network training
## Installation
Project is made with Poetry -> https://python-poetry.org/
First you need installed Poetry and then with pyproject.toml, it's simple to install dependencies:

```bash
poetry install
```


## Usage
To use the model, you need  OpenAI API key exported as global variable 

```bash
OPENAI_API_KEY = $API_KEY
```

And of course balance greater than 0.00$ on OpenAI account.

Run the following command:

```bash
poetry run python user_input.py
```

And type in file text, that includes your favourite text. Script will output decision and how confident it's about it.

If you would like to train NN by yourself, feel free to modify and use train.py script :

```bash
poetry run python train.py
```

Model will be saved as "model" - remember to change path of torch.load() in user_input (path is set to "model") or change it to "base_model_params"

## Dataset
https://huggingface.co/datasets/andythetechnerd03/AI-human-text

## Model Architecture
The model uses a deep neural network architecture with multiple layers to effectively capture the complexities of emotions and sentiments in song lyrics. It includes:
- Flatten layer for input
- Fully connected Linear layers for classification

## Results
The model has been trained on a diverse dataset of LLM and human generated texts  and tested for accuracy and reliability using test dataset. 

Model came out with overall test accuracy measurment : 

- Accuracy: 99.2% 
- Avg. loss: 0.054887
- Precision: 0.9972
- Recall: 0.9823
- F1-score: 0.9897

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
- Thanks to the creators of OpenAI and PyTorch for their powerful frameworks.
- This is a hobby project and not intended for professional LLM detection. For more advanced detectors use these :
- https://www.scribbr.com/ai-detector/
- https://quillbot.com/ai-content-detector