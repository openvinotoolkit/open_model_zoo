import pickle as pkl
import json

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3


class Vocab(object):
    def __init__(self, vocab_path):
        if '.pkl' in vocab_path:
            with open(vocab_path, "rb") as f:
                vocab_dict = pkl.load(f)
        elif 'json' in vocab_path:
            with open(vocab_path, "r") as f:
                vocab_dict = json.load(f)
                for k, v in vocab_dict['id2sign'].items():
                    del vocab_dict['id2sign'][k]
                    vocab_dict['id2sign'][int(k)] = v
        else:
            raise ValueError("Wrong extension of the vocab file")
        self.id2sign = vocab_dict["id2sign"]

    def construct_phrase(self, indices):
        phrase_converted = []
        for token in indices:
            if token == END_TOKEN:
                break
            phrase_converted.append(
                self.id2sign.get(token, "?"))
        return " ".join(phrase_converted)
