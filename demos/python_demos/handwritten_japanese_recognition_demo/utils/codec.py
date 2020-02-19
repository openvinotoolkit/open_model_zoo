
import numpy as np

class CTCCodec(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
             # NOTE: 0 is reserved for 'blank' token required by CTCLoss
             self.dict[char] = i + 1

        # dummy '[blank]' token for CTCLoss (index 0)
        self.character = ['[blank]'] + dict_character

    def decode(self, preds):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        # Select max probabilty (greedy decoding) then decode index to character
        preds_index = np.argmax(preds, 2)
        preds_index = preds_index.transpose(1,0)
        preds_index_reshape = preds_index.reshape(-1)
        preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

        text_index = preds_index_reshape
        length = preds_sizes

        for l in length:
            t = text_index[index:index + l]

            # NOTE: t might be zero size
            if t.shape[0] == 0:
                continue

            char_list = []
            for i in range(l):
                # removing repeated characters and blank.
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts
