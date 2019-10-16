import numpy as np

def ctc_beam_search_decoder(input_tensor, text_label, blank, beamwidth):
    pred = input_tensor.squeeze()

    t_step = pred.shape[0]
    idx_b = text_label.index(blank)

    _pB = {}
    _pNB = {}
    _pT = {}

    _init = () # init state, to make sure the first index is not blank ****

    for __t in ['c', 'l']:
        _pB[__t] = {}
        _pNB[__t] = {}
        _pT[__t] = {}

    _pB['l'][_init] = 1
    _pNB['l'][_init] = 0
    _pT['l'][_init] = 1

    for _t in range(t_step):
        _pB['c'] = {}
        _pNB['c'] = {}
        _pT['c'] = {}

        for _candidate in _pNB['l']:
            _TpNB = 0
            if _candidate != _init:
                _TpNB = _pNB['l'][_candidate] * pred[_t][_candidate[-1]]
            _TpB = _pT['l'][_candidate] * pred[_t][idx_b]
            if _candidate in _pNB['c']:
                _pNB['c'][_candidate] += _TpNB
            else:
                _pNB['c'][_candidate] = _TpNB
            _pB['c'][_candidate] = _TpB
            _pT['c'][_candidate] = _pNB['c'][_candidate] + _pB['c'][_candidate]

            for i, v in np.ndenumerate(pred[_t]):
                if i < (idx_b,):
                    extand_t = _candidate + (i,)
                    if len(_candidate) > 0 and _candidate[-1] == i:
                        _TpNB = v * _pB['l'][_candidate]

                    else:
                        _TpNB = v * _pT['l'][_candidate]

                    if extand_t in _pT['c']:
                        _pT['c'][extand_t] += _TpNB
                        _pNB['c'][extand_t] += _TpNB
                    else:
                        _pB['c'][extand_t] = 0
                        _pT['c'][extand_t] = _TpNB
                        _pNB['c'][extand_t] = _TpNB

        sorted_c = sorted(_pT['c'].items(), reverse=True, key=lambda item:item[1])
        _pB['l'] = {}
        _pNB['l'] = {}
        _pT['l'] = {}
        for _sent in sorted_c[:beamwidth]:
            _pB['l'][_sent[0]] = _pB['c'][_sent[0]]
            _pNB['l'][_sent[0]] = _pNB['c'][_sent[0]]
            _pT['l'][_sent[0]] = _pT['c'][_sent[0]]

    res = sorted(_pT['l'].items(), reverse=True, key=lambda item:item[1])[0]

    text = ''
    for idx, _r in enumerate(res[0]):
        text += text_label[_r[0]]

    return text
