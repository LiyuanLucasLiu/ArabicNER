import torch
import string
import numpy as np
import torch.nn as nn

def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def identify_cross_code(word):
    if is_english(word):
        return False
    if not is_english(word[0]) and not is_english(word[-1]):
        return False
    if not is_english(word[0]):
        ind = len(word) - 1
        while ind >= 0:
            if not is_english(word[ind:]):
                if not word[ind+1:].isdigit():
                    return True
                else:
                    return False
            ind -= 1
    else:
        ind = 1
        while ind <= len(word) - 1:
            if not is_english(word[0:ind]):
                if not word[0: ind - 1].isdigit():
                    return True
                else:
                    return False
            ind += 1
    return False
    
class VariationalDropout(torch.nn.Module):

    def __init__(self, p=0.5):
        super(VariationalDropout, self).__init__()
        self.droprate = p

    def forward(self, x):
        if not self.training or not self.droprate:
            return x

        mask = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.droprate)
        mask = torch.autograd.Variable(mask, requires_grad=False) / (1 - self.droprate)
        mask = mask.expand_as(x)
        return mask * x

class WordDropout(torch.nn.Module):

    def __init__(self, p=0.05):
        super(WordDropout, self).__init__()
        self.droprate = p

    def forward(self, x):
        if not self.training or not self.droprate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.droprate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

def iob_iobes(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.startswith('O'):
            new_tags.append(tag)
        elif tag.startswith('B-'):
            if i + 1 < len(tags) and tags[i + 1].startswith('I-'):
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        else:
            assert tag.startswith('I-')
            if i + 1 < len(tags) and tags[i + 1].startswith('I-'):
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))

    return new_tags

def repackage_hidden(h, device = None):
    """
    Wraps hidden states in new Variables, to detach them from their history

    Parameters
    ----------
    h : ``Tuple`` or ``Tensors``, required.
        Tuple or Tensors, hidden states.

    Returns
    -------
    hidden: ``Tuple`` or ``Tensors``.
        detached hidden states
    """
    if type(h) == torch.Tensor:
        if device is None:
            return h.detach()
        else:
            return h.detach().to(device)
    else:
        return tuple(repackage_hidden(v) for v in h)
        
def init_linear(input_linear):
    """
    random initialize linear projection.
    """
    # bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    # nn.init.uniform_(input_linear.weight, -bias, bias)
    # if input_linear.bias is not None:
    #     input_linear.bias.data.zero_()
    return

def log_sum_exp(vec):
    max_score, _ = torch.max(vec, 1)

    return max_score + torch.log(torch.sum(torch.exp(vec - max_score.unsqueeze(1).expand_as(vec)), 1))

def adjust_learning_rate(optimizer, lr):
    """
    adjust learning to the the new value.
    Parameters
    ----------
    optimizer : required.
        pytorch optimizer.
    float :  ``float``, required.
        the target learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def rank_by_number(x_input):
    res = [tup.split(',') for tup in x_input.split('\n') if tup and not tup.isspace()]
    res.sort(key=lambda t: int(t[1]))
    res = list(filter(lambda t: not t[-1][-2].isdigit() or not t[-1][1].isdigit(), res))
    res = [','.join(tup) for tup in res]
    return '\n'.join(res)


def conlidate_code_switch(file_name, sind, eind, tp, surface_name):

    surface_name_cat = ' '.join(surface_name)
    if not identify_cross_code(surface_name_cat):
        return conlidate_punctuation(file_name, sind, eind, tp, surface_name)
    else:
        # print([file_name, str(sind), str(eind), tp, '1.0', surface_name_cat])
        cursor = 0
        results = ""
        while cursor < len(surface_name):
            if is_english(surface_name[cursor]) or identify_cross_code(surface_name[cursor]):
                end_of_span = cursor + 1
                while end_of_span < len(surface_name) and (is_english(surface_name[end_of_span]) or identify_cross_code(surface_name[end_of_span])):
                    end_of_span += 1
                results += conlidate_punctuation(file_name, sind + cursor, sind + end_of_span - 1, tp, surface_name[cursor: end_of_span])
                cursor = end_of_span
            else:
                cursor += 1
        return results

def is_punctuation(word):
    return word != ''.join([char for char in word if char not in string.punctuation]) 

def conlidate_punctuation(file_name, sind, eind, tp, surface_name):

    cursor = 0
    while cursor < len(surface_name) and is_punctuation(surface_name[cursor]):
        cursor += 1

    end_cursor = len(surface_name) - 1
    while end_cursor >= 0 and is_punctuation(surface_name[end_cursor]):
        end_cursor -= 1

    if cursor <= end_cursor:
        surface_name_cat = ' '.join(surface_name[cursor: end_cursor+1])
        return ','.join([file_name, str(sind+cursor), str(sind+end_cursor), tp, '1.0', '"'+surface_name_cat.replace(',', '<comma>').replace('"', '<qu>') +'"']) + '\n'
    else:
        return ''
        