# Chatbot using TFlearn and NLTK

This repository contains a simple chatbot implemented using TFlearn and NLTK for natural language processing. The chatbot is trained on a dataset of intents to understand and respond to user input.

## Requirements

- Python 3.x
- TFlearn
- NLTK
- json
- numpy
- Tensorflow
- pickle

Install the required libraries using the following command:

```
pip install tflearn nltk 
```

You can download all the required libraries from pip command.

## Usage
1. Clone the repository:
```
git clone https://github.com/crocks4123/AI-chatbot-basic.git
```
2. Navigate to the project directory:
```
cd AI-chatbot-basic
```
3. Run the chatbot via this python file:

```
python ai-chatbot.py
```
*Interact with the chatbot by entering text prompts. Type _"exit"_ to quit.


## Intents Data
The chatbot is trained on intents defined in the _bot-intents.json_ file. Each intent includes patterns, responses, and tags. You can also use _bot-intents-2.json_ file as well.

## Acknowledgements
* The chatbot is built using TFlearn and NLTK.
* The training data is provided in the bot-intents.json file.
* Feel free to explore, modify, and enhance the chatbot based on your requirements!
