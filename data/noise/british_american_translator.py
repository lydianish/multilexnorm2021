import requests, json


class BritishAmericanTranslator:
    def __init__(self):
#        url = "https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/british_spellings.json"
#        self.british_to_american_dict = requests.get(url).json()
        with open("data/noise/british_spellings.json") as f:
            self.british_to_american_dict = json.load(f)
#        url = "https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json"
#        self.american_to_british_dict = requests.get(url).json()
        with open("data/noise/american_spellings.json") as f:
            self.american_to_british_dict = json.load(f)

    def translate(self, sentence, dictionary):
        out_sentence = sentence.copy()
        for i, word in enumerate(sentence):
            if word in dictionary:
                out_sentence[i] = dictionary[word]
        return out_sentence

    def to_american(self, sentence):
        return self.translate(sentence, self.british_to_american_dict)

    def to_british(self, sentence):
        return self.translate(sentence, self.american_to_british_dict)
