import numpy as np


class CTCCodec(object):
    """ Convert between text-label and text-index """
    def __init__(self, characters, designated_characters, top_k):
        # characters (str): set of the possible characters.
        self.designated_character_list = None
        if designated_characters != None:
            with open(designated_characters, encoding='utf-8') as f:
                self.designated_character_list = [line.strip() for line in f]

        self.top_k = top_k
        dict_character = list(characters)

        self.dict = {}
        for i, char in enumerate(dict_character):
             # NOTE: 0 is reserved for 'blank' token required by CTCLoss
             self.dict[char] = i + 1

        # dummy '[blank]' token for CTCLoss (index 0)
        self.characters = ['[blank]'] + dict_character

    def decode(self, preds):
        """ convert text-index into text-label. """
        texts = []
        # Select max probabilty (greedy decoding) then decode index to character
        preds_index = np.argmax(preds, 2) # WBD - > WB
        preds_index = preds_index.transpose(1, 0) # WB -> BW
        preds_index_reshape = preds_index.reshape(-1) # B*W

        char_list = []
        if self.designated_character_list != None:
            # Store the top k indices in each time step in a 2D matrix
            preds_index_filter = preds.transpose(1, 0, 2) # WBD -> BWD  B=1
            preds_index_filter = np.squeeze(preds_index_filter) # WD
            preds_top_k_index_matrix = np.zeros((preds_index_filter.shape[0], self.top_k))
            for i in range(preds_index_filter.shape[0]):
                row = preds_index_filter[i, :]
                top_k_idx = row.argsort()[::-1][0:self.top_k]
                preds_top_k_index_matrix[i, :] = top_k_idx

            for i in range(len(preds_index_reshape)):
                if preds_index_reshape[i] != 0 and (not (i > 0 and preds_index_reshape[i - 1] == preds_index_reshape[i])):
                    append_char = self.characters[preds_index_reshape[i]]
                    # Traverse the top k index array until a designated character is found
                    if not append_char in self.designated_character_list:
                        for index in preds_top_k_index_matrix[i, :]:
                            if self.characters[int(index)] in self.designated_character_list:
                                append_char = self.characters[int(index)]
                                break
                    char_list.append(append_char)
        else:
            for i in range(len(preds_index_reshape)):
                if preds_index_reshape[i] != 0 and (not (i > 0 and preds_index_reshape[i - 1] == preds_index_reshape[i])):
                    char_list.append(self.characters[preds_index_reshape[i]])

        text = ''.join(char_list)
        texts.append(text)

        return texts
