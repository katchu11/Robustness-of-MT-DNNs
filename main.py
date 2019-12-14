# Usage: python3 main.py --mode dev --model bilstm \
# --load tmp/lstm_epochs=4 --num_examples 100
#import sys
#sys.path.insert(1,"/home/kashyap/mt-dnn/mt_dnn")

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import spacy
from spell_checkers.atd_checker import ATDChecker
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from collections import defaultdict
import time
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import datetime
import biLstm_with_chars # word + char model biLSTM model
import biLstm_char_only  # char only biLSTM model
import biLstm
from biLstm_with_chars import BiLSTM
# from CNN import CNN
# from RNN_with_char import RNN
from spellchecker import SpellChecker

#from model import MTDNNModel
#from data_utils.glue_utils import submit, eval_model
import torch
import os
#from mt_dnn.batcher import BatchGen
import pickle

from mt_dnn.model import MTDNNModel
from experiments.glue.glue_utils import submit, eval_model
import torch
import os
from mt_dnn.batcher import BatchGen
import pickle

import argparse
from random import shuffle
import sys
import string
from nltk.corpus import stopwords as SW
#import hunspell_checker
#from hunspell_checker import HunspellChecker
import attacks
from tqdm import tqdm
import pickle
import pandas as pd
import dynet_config
dynet_config.set(random_seed=42)
import dynet as dy
dyparams = dy.DynetParams()
dyparams.set_mem(8000)
dynet_config.set_gpu()
import numpy as np
np.random.seed(42)
import random
random.seed(42)

sys.path.insert(0, 'defenses/scRNN/')
sys.path.append('spell_checkers/')
nlp = English()
spell = SpellChecker()
from corrector import ScRNNChecker

# personal logging lib
import log
log.DEBUG = True


stopwords = set(SW.words("english")) | set(string.punctuation)
# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
s2i = defaultdict(lambda: len(s2i))
c2i = defaultdict(lambda: len(c2i))
UNK = w2i["<unk>"]
CHAR_UNK = c2i["<unk>"]
NUM_EXAMPLES = 100

vocab_set = set()
char_vocab_set = set()
singles = False

def read_valid_lines_single(filename):
    """reads files (ignores the neutral reviews)

    Arguments:
        filename -- data file

    Returns:
        lines, tags: list of reviews, and their tags
    """
    if(singles):
        lines =[]
        tags = []
        filename = "data/canonical_data/sst_dev.tsv"
        df = pd.read_csv(filename, sep = "\t")
        lines = df.iloc[:,2].tolist()
        tags = df.iloc[:,1].tolist()
        tags = [1 if x=="neutral" else 0 for x in tags]
        return lines, tags
    else:
        premises =[]
        hypotheses = []
        tags = []
        filename = "data/canonical_data/scitail_dev.tsv"
        df = pd.read_csv(filename, sep = "\t")
        premises = df.iloc[:,2].tolist()
        hypotheses = df.iloc[:,3].tolist()
        tags = df.iloc[:,1].tolist()
        #tags = [1 if x=="neutral" else 0 if x=="contradiction" else 2 for x in tags]
        tags = [0 if x=="neutral" else 1 for x in tags]
        print(tags[:5])
        list_of_lists = []
        for i in range(len(premises)):
            list_of_lists.append([premises[i],hypotheses[i],tags[i]])
        return list_of_lists, None




    # with open(filename, encoding="utf8") as f:
    #     for line in f:
    #         blocks = line.strip().split('\t')
    #         lines.append(blocks[0])
    #         tags.append(blocks[2])
    # print(lines2[:5])
    # print(type(lines[0]))
    # print(tags[:5])
    # print(type(tags[0]))
    # print("starting to read %s" %(filename))
    #
    # lines, tags = [], []
    # with open(filename, 'r') as f:
    #     for line in f:
    #         tag, words = line.lower().strip().split(" ||| ")
    #         if tag == '0' or tag == '1': tag = '0'
    #         if tag == '3' or tag == '4': tag = '1'
    #         if tag == '2': continue
    #         tags.append(int(tag))
    #         lines.append(words)
    # print("SPACE")
    # print(lines[:5])
    # print(type(lines[0]))
    # print(tags[:5])
    # print(type(tags[0]))


def read_dataset(filename, drop=False, swap=False, key=False, add=False, all=False):
    """creates a dataset from reading reviews; uses word and tag dicts

    Arguments:
        filename  -- input file
    """

    lines, tags = read_valid_lines_single(filename)
    ans = []
    for line, tag in zip(lines, tags):
        words = [x for x in line.split(" ")]
        word_idxs = [w2i[x] for x in line.split(" ")]
        char_idxs = []
        for word in words: char_idxs.append([c2i[i] for i in word])
        tag = t2i[tag]
        ans.append((word_idxs, char_idxs, tag))
        if (drop or swap or key or add or all) and random.random() < char_drop_prob:
            if drop:
                line = drop_a_char(line)
            elif swap:
                line = swap_a_char(line)
            elif key:
                line = key_a_char(line)
            elif add:
                line = add_a_char(line)
            elif all:
                perturbation_fns = [drop_a_char, swap_a_char, add_a_char, swap_a_char]
                perturbation_fn = np.random.choice(perturbation_fns, 1)[0]
                line = perturbation_fn(line)

            words = [x for x in line.split(" ")]
            word_idxs = [w2i[x] for x in line.split(" ")]
            char_idxs = []
            for word in words: char_idxs.append([c2i[i] for i in word])
            ans.append((word_idxs, char_idxs, tag))
    return ans



def normalize(x):
    """ normalizes the scores in x, works only for 1D """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def create_vocabulary(filename):
    """given a file, creates the vocab set from its words

    Arguments:
        filename -- input file
    """

    global vocab_set
    lines, _ = read_valid_lines_single(filename)
    for line in lines:
        for word in line[0].split(" "):
            vocab_set.add(word)
            for char in word:
                char_vocab_set.add(char)
        for word in line[1].split(" "):
            vocab_set.add(word)
            for char in word:
                char_vocab_set.add(char)
    return

def get_word_and_char_indices_adv(line):
    pickle_off = open("tokenizer.pkl", "rb")
    tokenizer = pickle.load(pickle_off)
    premise = tokenizer.tokenize(line)
    if len(premise) > 512 - 3: premise = premise[:512 - 3]
    input_ids = tokenizer.convert_tokens_to_ids(
        ['[CLS]'] + premise + ['[SEP]'])
    type_ids = [0] * (len(premise) + 2)
    features = {'uid': 0, 'label': 0,
                'token_id': input_ids, 'type_id': type_ids}

    return [features], None
def get_word_and_char_indices(line):
    if(singles):
        pickle_off = open("tokenizer.pkl", "rb")
        tokenizer = pickle.load(pickle_off)
        premise = tokenizer.tokenize(line)
        if len(premise) > 512 - 3: premise = premise[:512 - 3]
        input_ids = tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + premise + ['[SEP]'])
        type_ids = [0] * (len(premise) + 2)
        features = {'uid': 0, 'label': 0,
                    'token_id': input_ids, 'type_id': type_ids}

        return [features], None
    else:
        pickle_off = open("tokenizer.pkl", "rb")
        tokenizer = pickle.load(pickle_off)
        premise = tokenizer.tokenize(line[0])
        hypothesis = tokenizer.tokenize(line[1])
        if len(premise) > 512 - 3: premise = premise[:512 - 3]
        input_ids = tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + premise + ['[SEP]'] + hypothesis +  ['[SEP]'])
        type_ids = [0] * (len(premise) + 2) + [1] * (len(hypothesis) + 1)
        features = {'uid': 0, 'label': 0,
                    'token_id': input_ids, 'type_id': type_ids}

        return [features], None


def check_against_spell_mistakes(filename):
    if(singles):
        lines, tags = read_valid_lines_single(filename)
        spell_check_model = BertForMaskedLM.from_pretrained("bert-base-uncased", cache_dir="/data/kashyap_data/")
        spell_check_model.eval()
        spell_check_model.to('cuda')
        c = list(zip(lines, tags))
        random.shuffle(c)
        lines, tags = zip(*c)
        lines = lines
        tags = tags

        # if in small (or COMPUTATION HEAVY) modes
        if params['small']:
            lines = lines[:200]
            tags = tags[:200]
        if params['small'] and params['sc_atd']:
            lines = lines[:99]
            tags = tags[:99]

        inc_count = 0.0
        inc_count_per_attack = [0.0 for _ in range(NUM_ATTACKS+1)]
        error_analyser = {}
        for line, tag in tqdm(zip(lines, tags)):

            w_i, c_i = get_word_and_char_indices(line)
            if params['is_spell_check']:
                """
                pickle_off = open("tokenizer.pkl", "rb")
                bert_tokenizer = pickle.load(pickle_off)
                tokenizer = nlp.Defaults.create_tokenizer(nlp)
                tokens = [t.text for t in tokenizer(line)]
                misspelled = spell.unknown(tokens)
                indices = []
                for i in misspelled:
                    indices.append(tokens.index(i))
                    tokens[tokens.index(i)] = "[MASK]"
                tokens.insert(0, "[CLS]")
                tokens.insert(-1, "[SEP]")
                new_string = " ".join(tokens)
                bert_tokenized_text = bert_tokenizer.tokenize(new_string)
                indexed_tokens = bert_tokenizer.convert_tokens_to_ids(
                    bert_tokenized_text)
                segments_ids = [0]*len(bert_tokenized_text)

                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segments_ids])
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensors = segments_tensors.to('cuda')
                with torch.no_grad():
                    outputs = spell_check_model(tokens_tensor, token_type_ids=segments_tensors)
                    predictions = outputs[0]
                predicted_tokens = []
                indices = [i+1 for i in indices]
                for i in indices:
                    predicted_tokens.append(bert_tokenizer.convert_ids_to_tokens(
                        [torch.argmax(predictions[0, i]).item()])[0])
                for j in range(len(predicted_tokens)):
                    tokens[indices[j]] = predicted_tokens[j]
                tokens.remove("[SEP]")
                tokens.remove("[CLS]")
                final_string = " ".join(tokens)
                """
                temp = line.split(" ")
                new_string = []
                for i in temp:
                    new_string.append(spell.correction(i))
                temp = " ".join(new_string)
                line = temp
                w_i, c_i = get_word_and_char_indices(line)

            # check if model prediction is incorrect, if yes, continue
            model_prediction = predict(w_i, c_i)
            if tag != model_prediction:
                # already incorrect, no attack needed
                inc_count += 1
                inc_count_per_attack[0] += 1.0
                continue

            found_incorrect = False

            worst_example = line
            worst_confidence = 1.0
            worst_idx = -1
            ignore_incides=set()

            for attack_count in range(1, 1 + NUM_ATTACKS):

                ignore_incides.add(worst_idx)

                if 'drop' in type_of_attack:
                    gen_attacks = attacks.drop_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
                elif 'swap' in type_of_attack:
                    gen_attacks = attacks.swap_one_attack(worst_example, include_ends=params['include_ends'])
                elif 'key' in type_of_attack:
                    gen_attacks = attacks.key_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
                elif 'add' in type_of_attack:
                    gen_attacks = attacks.add_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
                elif 'all' in type_of_attack:
                    gen_attacks = attacks.all_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
                for idx, adversary in gen_attacks:
                    original_adv = adversary
                    if found_incorrect: break
                    if params['is_spell_check']:
                        """
                        pickle_off = open("tokenizer.pkl", "rb")
                        bert_tokenizer = pickle.load(pickle_off)
                        tokenizer = nlp.Defaults.create_tokenizer(nlp)
                        tokens = [t.text for t in tokenizer(line)]
                        misspelled = spell.unknown(tokens)
                        indices = []
                        for i in misspelled:
                            indices.append(tokens.index(i))
                            tokens[tokens.index(i)] = "[MASK]"
                        tokens.insert(0, "[CLS]")
                        tokens.insert(-1, "[SEP]")
                        new_string = " ".join(tokens)
                        bert_tokenized_text = bert_tokenizer.tokenize(new_string)
                        indexed_tokens = bert_tokenizer.convert_tokens_to_ids(
                            bert_tokenized_text)
                        segments_ids = [0]*len(bert_tokenized_text)

                        tokens_tensor = torch.tensor([indexed_tokens])
                        segments_tensors = torch.tensor([segments_ids])
                        tokens_tensor = tokens_tensor.to('cuda')
                        segments_tensors = segments_tensors.to('cuda')
                        with torch.no_grad():
                            outputs = spell_check_model(
                                tokens_tensor, token_type_ids=segments_tensors)
                            predictions = outputs[0]
                        predicted_tokens = []
                        indices = [i+1 for i in indices]
                        for i in indices:
                            predicted_tokens.append(bert_tokenizer.convert_ids_to_tokens(
                                [torch.argmax(predictions[0, i]).item()])[0])
                        for j in range(len(predicted_tokens)):
                            tokens[indices[j]] = predicted_tokens[j]
                        tokens.remove("[SEP]")
                        tokens.remove("[CLS]")
                        final_string = " ".join(tokens)
                        """
                        temp = line[0].split(" ")
                        new_string = []
                        for i in temp:
                            new_string.append(spell.correction(i))
                        temp = " ".join(new_string)
                        adversary = temp
                    w_i, c_i = get_word_and_char_indices(adversary)
                    adv_pred = predict(w_i, c_i)
                    confidence = get_confidence(w_i, c_i)

                    if confidence < worst_confidence:
                        worst_confidence = confidence
                        worst_idx = idx
                        worst_example = adversary

                    if adv_pred != tag:
                        # found incorrect prediction
                        found_incorrect = True
                        break

                if found_incorrect:
                    inc_count += 1.0
                    inc_count_per_attack[attack_count] += 1.0
                    if params['analyse']:
                        error_analyser[line] = {}
                        error_analyser[line]['adversary'] = original_adv.split()[idx]
                        error_analyser[line]['correction'] = adversary.split()[idx]
                        error_analyser[line]['idx'] = idx

                    break
    else:
        lines, tags = read_valid_lines_single(filename)
        spell_check_model = BertForMaskedLM.from_pretrained("bert-base-uncased", cache_dir="/data/kashyap_data/")
        spell_check_model.eval()
        spell_check_model.to('cuda')
        # c = list(zip(lines, tags))
        # random.shuffle(c)
        # lines, tags = zip(*c)
        lines = lines
        tags = tags

        # if in small (or COMPUTATION HEAVY) modes
        if params['small']:
            lines = lines[:200]
            tags = tags[:200]
        if params['small'] and params['sc_atd']:
            lines = lines[:99]
            tags = tags[:99]

        inc_count = 0.0
        inc_count_per_attack = [0.0 for _ in range(NUM_ATTACKS+1)]
        error_analyser = {}
        for line in tqdm(lines):
            tag = line[2]
            w_i, c_i = get_word_and_char_indices(line)
            if params['is_spell_check']:
                temp = line[0].split(" ")
                new_string = []
                for i in temp:
                    new_string.append(spell.correction(i))
                temp = " ".join(new_string)
                line[0] = temp
                w_i, c_i = get_word_and_char_indices(line)

            # check if model prediction is incorrect, if yes, continue
            model_prediction = predict(w_i, c_i)
            if tag != model_prediction:
                # already incorrect, no attack needed
                inc_count += 1
                inc_count_per_attack[0] += 1.0
                continue

            found_incorrect = False

            worst_example = line[0] #could make random choice
            worst_confidence = 1.0
            worst_idx = -1
            ignore_incides=set()

            for attack_count in range(1, 1 + NUM_ATTACKS):

                ignore_incides.add(worst_idx)

                if 'drop' in type_of_attack:
                    gen_attacks = attacks.drop_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
                elif 'swap' in type_of_attack:
                    gen_attacks = attacks.swap_one_attack(worst_example, include_ends=params['include_ends'])
                elif 'key' in type_of_attack:
                    gen_attacks = attacks.key_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
                elif 'add' in type_of_attack:
                    gen_attacks = attacks.add_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
                elif 'all' in type_of_attack:
                    gen_attacks = attacks.all_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
                for idx, adversary in gen_attacks:
                    original_adv = adversary
                    if found_incorrect: break
                    if params['is_spell_check']:
                        temp = adversary.split(" ")
                        new_string = []
                        for i in temp:
                            new_string.append(spell.correction(i))
                        temp = " ".join(new_string)

                        adversary = adversary
                    w_i, c_i = get_word_and_char_indices_adv(adversary)
                    adv_pred = predict(w_i, c_i)
                    confidence = get_confidence(w_i, c_i)

                    if confidence < worst_confidence:
                        worst_confidence = confidence
                        worst_idx = idx
                        worst_example = adversary

                    if adv_pred != tag:
                        # found incorrect prediction
                        found_incorrect = True
                        break

                if found_incorrect:
                    inc_count += 1.0
                    inc_count_per_attack[attack_count] += 1.0
                    if params['analyse']:
                        error_analyser[line] = {}
                        error_analyser[line]['adversary'] = original_adv.split()[idx]
                        error_analyser[line]['correction'] = adversary.split()[idx]
                        error_analyser[line]['idx'] = idx

                    break

    for num in range(NUM_ATTACKS + 1):
        log.pr_red('adversarial accuracy of the model after %d attacks = %.2f'
                %(num, 100. * (1 - sum(inc_count_per_attack[:num+1])/len(lines))))

    if params['analyse']:
        curr_time = datetime.datetime.now().strftime("%B_%d_%I:%M%p")
        pickle.dump(error_analyser, open("error_analyser_" + str(curr_time) + ".p", 'wb'))

    return None


# make argparse
parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--load', dest='input_file', type=str, default="",
        help = 'load already trained model')
parser.add_argument('--save', dest='output_file', type=str, default="",
        help = 'save existing model')
parser.add_argument('--model', dest='model_type', type=str, default="lstm",
        help = 'architecture of the model: lstm or rnn or cnn')
parser.add_argument('--mode', dest='mode', type=str, default="dev",
        help = 'training or dev?')
parser.add_argument('--attack', dest='type_of_attack', type=str, default=None,
        help='type of attack you want, swap/drop/add/key/all')
parser.add_argument('--small', dest='small', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')

parser.add_argument('--defense', dest='is_spell_check', action='store_true')
parser.add_argument('--sc-neutral', dest='unk_output', action='store_true')
parser.add_argument('--sc-background', dest='sc_background', action='store_true')
parser.add_argument('--analyse', dest='analyse', action='store_true')
parser.set_defaults(is_spell_check=False)

parser.add_argument('--include-ends', dest='include_ends', action='store_true')
parser.set_defaults(include_ends=False)

# data augmentation flags
parser.add_argument('--da-drop', dest='da_drop', action='store_true')
parser.add_argument('--da-key', dest='da_key', action='store_true')
parser.add_argument('--da-add', dest='da_add', action='store_true')
parser.add_argument('--da-swap', dest='da_swap', action='store_true')
parser.add_argument('--da-all', dest='da_all', action='store_true')
parser.add_argument('--da-drop-prob', dest='da_drop_prob', type=float, default=0.5)

parser.add_argument('--num-attacks', dest='num_attacks', type=int, default=0)
parser.add_argument('--dynet-seed', dest='dynet-seed', type=int, default=42)

# adversarial training flags
parser.add_argument('--adv-drop', dest='adv_drop', action='store_true')
parser.add_argument('--adv-swap', dest='adv_swap', action='store_true')
parser.add_argument('--adv-key', dest='adv_key', action='store_true')
parser.add_argument('--adv-add', dest='adv_add', action='store_true')
parser.add_argument('--adv-all', dest='adv_all', action='store_true')
parser.add_argument('--adv-prob', dest='adv_prob', type=float, default=0.1)

# model names for spell check models
parser.add_argument('--sc-model-path', dest='sc_model_path', type=str, default=None,
        help = 'the model path for ScRNN model')
parser.add_argument('--sc-model-path-bg', dest='sc_model_path_bg', type=str, default=None,
        help = 'the model path for ScRNN background model')
parser.add_argument('--sc-elmo', dest='sc_elmo', action='store_true')
parser.add_argument('--sc-elmo-bg', dest='sc_elmo_bg', action='store_true')
parser.add_argument('--sc-atd', dest='sc_atd', action='store_true')
parser.add_argument('--sc-vocab-size', dest='sc_vocab_size', type=int, default=9999)
parser.add_argument('--sc-vocab-size-bg', dest='sc_vocab_size_bg', type=int, default=78470)

parser.add_argument('--task-name', dest='task_name', type=str, default="")

params = vars(parser.parse_args())

# logging details
log.DEBUG = params['debug']

model_type = params['model_type']
input_file = params['input_file']
mode = params['mode']
type_of_attack = params['type_of_attack']
char_drop_prob = params['da_drop_prob']
NUM_ATTACKS = params['num_attacks']

SC_MODEL_PATH = params['sc_model_path']
SC_MODEL_PATH_BG = params['sc_model_path_bg']

if params['sc_atd']:
    checker = ATDChecker()

elif SC_MODEL_PATH_BG is None or params['sc_background']:
    # only foreground spell correct model...
    checker = ScRNNChecker(model_name=SC_MODEL_PATH, use_background=False,
            unk_output=params['unk_output'], use_elmo=params['sc_elmo'],
            task_name=params['task_name'], vocab_size=params['sc_vocab_size'],
            vocab_size_bg=params['sc_vocab_size_bg'])
else:
    checker = ScRNNChecker(model_name=SC_MODEL_PATH, model_name_bg=SC_MODEL_PATH_BG,
            use_background=True, unk_output=params['unk_output'],
            use_elmo=params['sc_elmo'], use_elmo_bg=params['sc_elmo_bg'],
            task_name=params['task_name'], vocab_size=params['sc_vocab_size'],
            vocab_size_bg=params['sc_vocab_size_bg'])

model = None
# train = read_dataset("data/classes/train.txt")

# modify the dicts so that they return unk for unseen words/chars
w2i = defaultdict(lambda: UNK, w2i)
c2i = defaultdict(lambda: CHAR_UNK, c2i)

# dev = read_dataset("data/classes/dev.txt")
# test = read_dataset("data/classes/test.txt")


def evaluate(filename="data/classes/test.txt"):
    if(singles):
        lines, tags = read_valid_lines_single(filename) #[premise,hypothesis], tags
        correct = 0.0
        for line, tag in tqdm(zip(lines, tags)):
            w_i, c_i = get_word_and_char_indices(line)
            pred = predict(w_i, c_i)
            if pred == tag: correct += 1.0
        log.pr_green("accuracy of the model on test set = %.4f [No spell checks]" % (correct / len(lines)))
        return
    else:
        lines, tags = read_valid_lines_single(filename)
        correct = 0.0
        for line in tqdm(lines):
            w_i, c_i = get_word_and_char_indices(line)
            pred = predict(w_i, c_i)
            if pred == line[2]:
                correct += 1.0 #[prem, hyp, tag]
        log.pr_green("accuracy of the model on test set = %.4f [No spell checks]" % (correct / len(lines)))
        return

def predict(words, chars):
    dev_data = BatchGen(words, is_train=False, gpu=True,
                       batch_size=1, maxlen=512)
    dev_metrics, dev_predictions, scores, golds, dev_ids = eval_model(
       model, dev_data, 0, use_cuda=True, with_label = False)
    return dev_predictions[0]

   #scores = model.calc_scores(words, chars)
   #pred = np.argmax(scores.npvalue())
   #return pred


def get_confidence(words, chars):
    dev_data = BatchGen(words, is_train=False, gpu=True,
                        batch_size=1, maxlen=512)
    dev_metrics, dev_predictions, scores, golds, dev_ids = eval_model(
       model, dev_data, 0, use_cuda=True, with_label = False)
    return max(scores)



def drop_a_char(sentence):
    words = sentence.split(" ")

    for _ in range(10):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) < 3: continue
        pos = random.randint(1, len(words[word_idx])-2)
        words[word_idx] = words[word_idx][:pos] + words[word_idx][pos+1:]
        sentence = " ".join(words)
        break
    return sentence

def swap_a_char(sentence):
    words = sentence.split(" ")
    for _ in range(100):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) <= 3: continue
        pos = random.randint(1, len(words[word_idx])-3)
        #words[word_idx] = words[word_idx][:pos] + words[word_idx][pos+1:]
        words[word_idx] = words[word_idx][:pos] + words[word_idx][pos:pos+2][::-1] + words[word_idx][pos+2:]
        sentence = " ".join(words)
        break
    return sentence

def key_a_char(sentence):
    words = sentence.split(" ")
    for _ in range(100):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) <= 3: continue
        pos = random.randint(1, len(words[word_idx])-2)
        neighboring_chars = attacks.get_keyboard_neighbors(words[word_idx][pos])
        random_neighbor = np.random.choice(neighboring_chars, 1)[0]
        words[word_idx] = words[word_idx][:pos] + random_neighbor + words[word_idx][pos+1:]
        sentence = " ".join(words)
        break
    return sentence

def add_a_char(sentence):
    words = sentence.split(" ")
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    alphabets = [i for i in alphabets]
    for _ in range(100):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) <= 3: continue
        pos = random.randint(1, len(words[word_idx])-1)
        #words[word_idx] = words[word_idx][:pos] + words[word_idx][pos+1:]
        new_char = np.random.choice(alphabets, 1)[0]
        words[word_idx] = words[word_idx][:pos] + new_char + words[word_idx][pos:]
        sentence = " ".join(words)
        break
    return sentence


def start_adversarial_training(trainer):
    lines, tags = read_valid_lines_single("data/classes/train.txt")
    train = [(lines[i], tags[i]) for i in range(len(lines))]
    for ITER in range(10):
        train_loss = 0.0
        train_correct = 0.0
        start = time.time()
        #TODO shuffle train
        random.shuffle(train)
        print("Length of training examples = %d" %(len(train)))
        for line, tag in train:
            w_i, c_i = get_word_and_char_indices(line)
            scores = model.calc_scores(w_i, c_i)
            my_loss = dy.pickneglogsoftmax(scores, t2i[tag])
            train_loss += my_loss.value()
            my_loss.backward()
            trainer.update()
            pred = np.argmax(scores.npvalue())
            if pred == t2i[tag]: train_correct += 1
        print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
        print("iter %r: train acc=%.4f" % (ITER, train_correct / len(train)))
        # Compute dev loss
        dev_loss = 0.0
        dev_correct = 0.0
        for words, chars, tag in dev:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            dev_loss += my_loss.value()
            pred = np.argmax(scores.npvalue())
            if pred == tag: dev_correct += 1
        print("iter %r: dev loss/sent=%.4f, time=%.2fs" % (ITER, dev_loss / len(dev), time.time() - start))
        print("iter %r: dev acc=%.4f" % (ITER, dev_correct / len(dev)))

        # compute test loss
        test_loss = 0.0
        test_correct = 0.0
        for words, chars, tag in test:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            test_loss += my_loss.value()
            pred = np.argmax(scores.npvalue())
            if pred == tag: test_correct += 1
        print("iter %r: test loss/sent=%.4f, time=%.2fs" % (ITER, test_loss / len(test), time.time() - start))
        print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))
        model.save("tmp/adv-" + model_type + "_drop=" + str(params['adv_drop']) + "_swap=" + str(params['adv_swap'])  + "_key" + str(params['adv_key']) + "_add" + str(params['adv_add']) + "_all" + str(params['adv_all']) + "_prob=" +  str(params['adv_prob']) +  "_epochs=" + str(ITER))

        if params['adv_swap'] or params['adv_drop'] or params['adv_key'] or params['adv_add'] or params['adv_all']:
            train.extend(add_more_examples(train,  params['adv_prob']/(ITER+2),
                        drop=params['adv_drop'], swap=params['adv_swap'],
                        key=params['adv_key'], add=params['adv_add'],
                        all=params['adv_all']))


def start_training(train, dev, trainer):
    if params['da_drop']:
        train = read_dataset("data/classes/train.txt", drop=True)
    if params['da_swap']:
        train = read_dataset("data/classes/train.txt", swap=True)
    if params['da_key']:
        train = read_dataset("data/classes/train.txt", key=True)
    if params['da_add']:
        train = read_dataset("data/classes/train.txt", add=True)
    if params['da_all']:
        train = read_dataset("data/classes/train.txt", all=True)
    for ITER in range(10):
        # Perform training
        random.shuffle(train)
        train_loss = 0.0
        start = time.time()
        train_correct = 0.0
        for words, chars, tag in train:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            train_loss += my_loss.value()
            my_loss.backward()
            trainer.update()
            pred = np.argmax(scores.npvalue())
            if pred == tag: train_correct += 1
        print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
        print("iter %r: train acc=%.4f" % (ITER, train_correct / len(train)))
        # Compute dev loss
        dev_loss = 0.0
        dev_correct = 0.0
        for words, chars, tag in dev:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            dev_loss += my_loss.value()
            pred = np.argmax(scores.npvalue())
            if pred == tag: dev_correct += 1
        print("iter %r: dev loss/sent=%.4f, time=%.2fs" % (ITER, dev_loss / len(dev), time.time() - start))
        print("iter %r: dev acc=%.4f" % (ITER, dev_correct / len(dev)))

        # compute test loss
        test_loss = 0.0
        test_correct = 0.0
        for words, chars, tag in test:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            test_loss += my_loss.value()
            pred = np.argmax(scores.npvalue())
            if pred == tag: test_correct += 1
        print("iter %r: test loss/sent=%.4f, time=%.2fs" % (ITER, test_loss / len(test), time.time() - start))
        print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))
        model.save("tmp/" + model_type + "_drop=" + str(params['da_drop']) + "_swap=" + str(params['da_swap']) + "_key=" + str(params['da_key']) + "_add=" + str(params['da_add']) + "_all=" + str(params['da_all'])  + "_prob=" +  str(char_drop_prob) +  "_epochs=" + str(ITER))


def get_qualitative_examples():
    lines, tags = read_valid_lines_single("data/classes/test.txt")
    c = list(zip(lines, tags))
    random.shuffle(c)
    lines, tags = zip(*c)
    lines = lines[:200]
    tags = tags[:200]

    for line, tag in tqdm(zip(lines, tags)):

        w_i, c_i = get_word_and_char_indices(line)

        # check if model prediction is incorrect, if yes, find next example...
        model_prediction = predict(w_i, c_i)
        if t2i[tag] != model_prediction:
            # already incorrect, not interesting...
            continue

        gen_attacks = attacks.all_one_attack(line)

        for idx, adversary in gen_attacks:
            adversary = checker.correct_string(adversary)
            w_i, c_i = get_word_and_char_indices(adversary)

            adv_pred = predict(w_i, c_i)

            if adv_pred == t2i[tag]:
                # this example doesn't break the model...
                continue

            corrected_string = checker.correct_string(adversary)
            w_i, c_i = get_word_and_char_indices(corrected_string)

            post_pred = predict(w_i, c_i)

            if post_pred != t2i[tag]:
                # after correction the tag isn't correct...
                continue

            log.pr(" -------------- ")
            log.pr("Original line = %s" %(line))
            log.pr("Original label = %s" %(tag))
            log.pr_red ("Adversary = %s" %(adversary))
            log.pr_green("Correction = %s" %(corrected_string))
            log.pr(" -------------- ")

    return None


def generate_ann():
    generate_dict = dict()
    lines, tags = read_valid_lines_single("data/classes/test.txt")
    c = list(zip(lines, tags))
    random.shuffle(c)
    lines, tags = zip(*c)
    lines = lines[:200]
    tags = tags[:200]

    # get the missclassified ones first
    missclassified_count = 0

    final_list = []

    for idx, line in enumerate(lines):
        if missclassified_count >= 50: break
        # need to attack the line...
        for _, adv in attacks.random_all_one_attack(line):
            w_i, c_i = get_word_and_char_indices(adv)
            model_prediction = predict(w_i, c_i)
            if model_prediction != t2i[tags[idx]]:
                # adversary found...
                final_list.append((line, tags[idx], 1))
                missclassified_count += 1


    for i in range(idx+1, idx+1+50):
        final_list.append((lines[i], tags[i], 0))

    pickle.dump(final_list, open("final_list_annotations.p, 'wb'"))

    for l in final_list:
        print (l[0].strip() + "\t" + str(l[1]) + "\t" + str(l[2]))


def add_more_examples(train, prob=0.1, drop=False, swap=False, key=False, add=False, all=False):
    extra_examples = []
    for line, tag in train:
        if random.random() > prob: continue # this is correct...
        if swap:
            gen_attacks = attacks.swap_one_attack(line)
        elif drop:
            gen_attacks = attacks.drop_one_attack(line)
        elif add:
            gen_attacks = attacks.add_one_attack(line)
        elif key:
            gen_attacks = attacks.key_one_attack(line)
        elif all:
            gen_attacks = attacks.all_one_attack(line)


        for _, adversary in gen_attacks:
                w_i, c_i = get_word_and_char_indices(adversary)
                adv_pred = predict(w_i, c_i)
                if adv_pred != t2i[tag]:
                    # found incorrect
                    extra_examples.append((adversary, tag))

    return extra_examples


def decode_tag(tag):
    return "POSITIVE" if tag == t2i['1'] else "NEGATIVE"


def main():
    # Read in the data
    global model

    nwords = len(w2i)
    ntags = len(t2i)
    nchars = len(c2i)

    if 'rnn' in model_type.lower():
        print ("Running a RNN model")
        model = RNN()
    elif 'cnn' in model_type.lower():
        print ("Running a CNN model")
        model = CNN()
    elif 'bilstm' == model_type.lower():
        print ("Running a BiLSTM char + word model ")
        model = biLstm_with_chars.BiLSTM()
    elif 'bilstm' in  model_type.lower() and 'word' in model_type.lower():
        print ("Running a BiLSTM word only model ")
        model = biLstm.BiLSTM()
    elif 'bilstm' in  model_type.lower() and 'char' in model_type.lower():
        print ("Running a BiLSTM char only model ")
        model = biLstm_char_only.BiLSTM()

    opt = {'log_file': 'checkpoints/scitail_tl_adamax_answer_opt0_gc0_ggc1_7_2_19/log.log', 'init_checkpoint': '/data/kashyap_data/mt_dnn_models/mt_dnn_large_uncased.pt', 'data_dir': 'data/domain_adaptation', 'data_sort_on': False, 'name': 'farmer', 'train_datasets': ['scitail'], 'test_datasets': ['scitail'], 'pw_tasks': ['qnnli'], 'update_bert_opt': 0, 'multi_gpu_on': False, 'mem_cum_type': 'simple', 'answer_num_turn': 5, 'answer_mem_drop_p': 0.1, 'answer_att_hidden_size': 128, 'answer_att_type': 'bilinear', 'answer_rnn_type': 'gru', 'answer_sum_att_type': 'bilinear', 'answer_merge_opt': 1, 'answer_mem_type': 1, 'answer_dropout_p': 0.1, 'answer_weight_norm_on': False, 'dump_state_on': False, 'answer_opt': [
        0], 'label_size': '2', 'mtl_opt': 0, 'ratio': 0, 'mix_opt': 0, 'max_seq_len': 512, 'init_ratio': 1, 'cuda': True, 'log_per_updates': 500, 'epochs': 5, 'batch_size': 16, 'batch_size_eval': 8, 'optimizer': 'adamax', 'grad_clipping': 0.0, 'global_grad_clipping': 1.0, 'weight_decay': 0, 'learning_rate': 5e-05, 'momentum': 0, 'warmup': 0.1, 'warmup_schedule': 'warmup_linear', 'vb_dropout': True, 'dropout_p': 0.1, 'dropout_w': 0.0, 'bert_dropout_p': 0.1, 'ema_opt': 0, 'ema_gamma': 0.995, 'have_lr_scheduler': True, 'multi_step_lr': '10,20,30', 'freeze_layers': -1, 'embedding_opt': 0, 'lr_gamma': 0.5, 'bert_l2norm': 0.0, 'scheduler_type': 'ms', 'output_dir': 'checkpoints/scitail_tl_adamax_answer_opt0_gc0_ggc1_7_2_19', 'seed': 2018, 'task_config_path': 'configs/tasks_config.json', 'tasks_dropout_p': [0.1]}
    state_dict = torch.load(
        "checkpoint/scitail_model_0.pt")
    config = state_dict['config']
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    opt.update(config)
    model = MTDNNModel(opt, state_dict=state_dict, num_train_step=50)

    print ("building vocabulary...")
    create_vocabulary('data/classes/train.txt')
    print ("done building vocabulary...")
    print ('size of the character vocab %s' %(len(char_vocab_set)))
   # trainer = model.build_model(nwords, nchars, ntags)

    # if input_file != "":
    #     model.load(input_file)

    if 'train' in mode.lower():
        if params['adv_swap'] or params['adv_drop'] or params['adv_key'] \
            or params['adv_add'] or params['adv_all']:
            start_adversarial_training(trainer)
        else:
            start_training(train, dev, trainer)
    elif 'gen' in mode.lower():
        generate_ann()
    elif 'examples' in mode.lower():
        get_qualitative_examples()
    else:
        evaluate()
        if type_of_attack is not None:
            check_against_spell_mistakes('data/classes/test.txt')

main()
