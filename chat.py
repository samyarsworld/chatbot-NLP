import torch
from model import BotModel
import random
import json
from pathlib import Path
from utils import bag_of_words, tokenize


MODEL_PATH = Path("model.pt")   
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data = torch.load(MODEL_PATH)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_tokens = data['all_tokens']
tags = data['tags']
model_state = data["model_state"]

# Get model
model = BotModel(input_size, hidden_size, output_size).to(DEVICE)

if MODEL_PATH.exists():
    model.load_state_dict(model_state)
else:
    print("Chat bot is NOT present!")


with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)



model.eval()

bot_name = "Sam"
print("Let's chat! (type 'q' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "q":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_tokens)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(DEVICE)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")

