import random
import json
import nltk
import torch
import transformers
from gtts import gTTS
import speech_recognition as sr
import os
import playsound
import config
import pyjokes
from nltk.stem.porter import PorterStemmer
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class Prepare_Data():
    def __init__(self, json_file, ignore_words):
        self.json_file = json_file
        self.patterns = []
        self.all_words = []
        self.tags = []
        self.xy = []
        self.X_train = []
        self.y_train = []
        self.ignore_words = ignore_words
        self.stemmer = PorterStemmer()
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)
    def stem(self, word):
        return self.stemmer.stem(word.lower())
    def bag_of_words(self, tokenized_sentence, words):
        sentence_words = [self.stem(word) for word in tokenized_sentence]
        # initialize bag with 0 for each word
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] = 1
        return bag
    def load_json(self):
        with open(self.json_file, 'r') as file:
            self.intents = json.load(file)
        return self.intents
    @staticmethod
    def text_to_speech(text):
        print(text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("a.mp3")
        playsound.playsound("a.mp3")
        os.remove("a.mp3")
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.text_to_speech("listening...")
            audio = recognizer.listen(source)
            recognizer.pause_threshold = 1
        try:
            self.text = recognizer.recognize_google(audio)
            print(self.text)
        except Exception:
            self.text = "say that again.."
        return self.text
    def prs1(self):
        for intent in self.load_json()['intents']:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                w = self.tokenize(pattern)
                self.all_words.extend(w)
                self.xy.append((w, tag))
                pattern = pattern.lower()
                self.patterns.append(pattern)
        self.all_words = [self.stem(w) for w in self.all_words if w not in self.ignore_words]
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))
        for (pattern_sentence, tag) in self.xy:
            bag = self.bag_of_words(pattern_sentence, self.all_words)
            self.X_train.append(bag)
            label = self.tags.index(tag)
            self.y_train.append(label)
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        return self.tags, self.all_words, self.patterns, self.X_train, self.y_train
class ChatDataset(Dataset):
    def __init__(self):
        self.prepare = Prepare_Data(json_file, ignore_words)
        self.tags, self.all_words, self.patterns, self.X_train, self.y_train = self.prepare.prs1()
        self.n_samples = len(self.X_train)
        self.x_data = self.X_train
        self.y_data = self.y_train
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
class Train():
    def __init__(self):
        self.num_epochs = config.NUM_EPOCHS
        self.batch_size = config.BATCH_SIZE
        self.learning_rate = config.LEARNING_RATE
        self.input_size = len(X_train[0])
        self.hidden_size = config.HIDDEN_SIZE
        self.num_classes = len(tags)
        self.dataset = ChatDataset()
        self.train_loader = DataLoader(dataset=self.dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              num_workers=0)
        self.model = NeuralNet(self.input_size, self.hidden_size, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.prepare = Prepare_Data(json_file, ignore_words)
        self.tags, self.all_words,_,_,_ = self.prepare.prs1()
    def train(self):
        for epoch in range(self.num_epochs):
            global loss
            for (words, labels) in self.train_loader:
                words = words.to(config.DEVICE)
                labels = labels.to(dtype=torch.long).to(config.DEVICE)
                outputs = self.model(words)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        print(f'final loss: {loss.item():.4f}')
        data = {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.num_classes,
            "all_words": self.all_words,
            "tags": self.tags
        }
        if loss < 0.001:
            FILE = "data.pth"
            torch.save(data, FILE)
            print(f'training complete. file saved to {FILE}')
class ChatBot():
    def __init__(self):
        self.tools = Prepare_Data(json_file, ignore_words)
        self.speech_to_text = self.tools.speech_to_text
        self.text_to_speech = self.tools.text_to_speech
        self.intents = self.tools.load_json()
        #self.tags, self.all_words, self.patterns, self.X_train, self.y_train =
        self.tags = self.tools.tags
        self.tokenize = self.tools.tokenize
        self.bag_of_words = self.tools.bag_of_words
    def load_model(self, model_file):
        data = torch.load(model_file)
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        model_state = data["model_state"]
        tags = data["tags"]
        model = NeuralNet(input_size, hidden_size, output_size).to(config.DEVICE)
        model.load_state_dict(model_state)
        model.eval()
        return model, tags
    def chat(self):
        nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-large", pretrained=True)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        while True:
            sentence = self.speech_to_text()
            if any(i in sentence for i in ["ok quit", "quit", "shutup", "go home"]):
                r = ["have fun", "see you later", "ok bye"]
                self.text_to_speech(random.choice(r))
                quit()
            elif "joke" in sentence:
                joke = pyjokes.get_joke(language="en", category="all")
                res = joke
            if any(i in sentence for i in patterns):
                in_ = self.tokenize(sentence)
                X = self.bag_of_words(in_, all_words)
                X = X.reshape(1, X.shape[0])
                X = torch.from_numpy(X).to(config.DEVICE)
                model, tags = self.load_model(model_file)
                output = model(X)
                _, predicted = torch.max(output, dim=1)
                tag = tags[predicted.item()]
                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()]
                if prob.item() > 0.75:
                    for intent in self.intents['intents']:
                        if tag == intent['tag']:
                            res = random.choice(intent['responses'])
            else:
                res = "none"
            if any(i in res for i in ["none", "None"]):
                chat = nlp(transformers.Conversation(sentence), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >> ") + 6:].strip()
            self.text_to_speech(res)

if __name__ == '__main__':
    json_file = "myintents.json"
    ignore_words = ["?", "!"]
    prepare = Prepare_Data(json_file, ignore_words)
    tags, all_words, patterns, X_train, y_train = prepare.prs1()

    # for traaining uncomment
    #train = Train()
    #train.train()
    model_file = "data.pth"

    #chat
    chat_bot = ChatBot()
    chat_bot.chat()


