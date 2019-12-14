from mt_dnn.model import MTDNNModel
from experiments.glue.glue_utils import submit, eval_model
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from mt_dnn.batcher import BatchGen
import pickle

Description_A = "blue 14.1 in. HP laptop"
Description_B = "sapphire 14.1 inch HP tablet"

pickle_off = open("tokenizer.pkl", "rb")
tokenizer = pickle.load(pickle_off)
print(tokenizer.convert_ids_to_tokens([101, 100, 2450, 2024, 3173, 14555, 1012, 102, 100, 2308, 2024, 23581, 2096, 3173, 2000, 2175, 14555, 1012, 102]))

def make_prediction(Description_A, Description_B, model_path, USE_GPU=True):
    # Loading tokenized using a stored Pickle Object, as it is more reliable
    # In case you'd like to create the object, you can do so here: bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)

    pickle_off = open("tokenizer.pkl", "rb")
    tokenizer = pickle.load(pickle_off)

    # Tokenizes the words into format required by Bert ex: "I am playing" -> ["I","am","play","##ing"]
    hypothesis = tokenizer.tokenize(Description_A)

    #If sequence is too long it truncates it to ensure it fits into BERT's max seq len and changes the words into numbers
    if len(hypothesis) > 512 - 3:
        hypothesis = hypothesis[:512 - 3]
    input_ids = tokenizer.convert_tokens_to_ids(
        ['[CLS]'] + hypothesis + ['[SEP]'])

    #Determnines what sentence it's in, doesn't really matter for single sentence, but important for 2 sentence classification
    type_ids = [0] * (len(hypothesis) + 2)

    #Concatenates all the important labels into a dictionary
    #UID : id number (no importance) ; label: "ground truth" (no importance when making a prediction)
    # token_id: representation of words ; type_id: position within a sentence
    features = {'uid': 0, 'label': 0,
                'token_id': input_ids, 'type_id': type_ids}

    # Loads data into a BatchGen object which is needed for making a prediction, nothing needed to change here
    dev_data = BatchGen([features],
                        batch_size=8,
                        gpu=True, is_train=False,
                        task_id=0,
                        maxlen=512,
                        pairwise=False,
                        data_type=0,
                        task_type=0)

    # function to convert token ids back to words
    print(tokenizer.convert_ids_to_tokens([101, 100, 5208, 2024, 17662, 9119, 2096, 3173, 2000, 2175, 14555, 2044, 2074, 5983, 6265, 1012, 102, 100, 2308, 2024, 23581, 2096, 3173, 2000, 2175, 14555, 1012, 102]))

    #hyper parameters: whatever is necessary is added as variables at the top
    opt = {'init_checkpoint': model_path, 'data_dir': 'data/domain_adaptation', 'data_sort_on': False, 'name': 'farmer', 'train_datasets': ['sst'], 'test_datasets': ['sst'], 'pw_tasks': ['qnnli'], 'update_bert_opt': 0, 'multi_gpu_on': False, 'mem_cum_type': 'simple', 'answer_num_turn': 5, 'answer_mem_drop_p': 0.1, 'answer_att_hidden_size': 128, 'answer_att_type': 'bilinear', 'answer_rnn_type': 'gru', 'answer_sum_att_type': 'bilinear', 'answer_merge_opt': 1, 'answer_mem_type': 1, 'answer_dropout_p': 0.1, 'answer_weight_norm_on': False, 'dump_state_on': False, 'answer_opt': [
        0], 'label_size': '2', 'mtl_opt': 0, 'ratio': 0, 'mix_opt': 0, 'max_seq_len': 512, 'init_ratio': 1, 'cuda': USE_GPU, 'log_per_updates': 500, 'epochs': 5, 'batch_size': 32, 'batch_size_eval': 8, 'optimizer': 'adamax', 'grad_clipping': 0.0, 'global_grad_clipping': 1.0, 'weight_decay': 0, 'learning_rate': 5e-05, 'momentum': 0, 'warmup': 0.1, 'warmup_schedule': 'warmup_linear', 'vb_dropout': True, 'dropout_p': 0.1, 'dropout_w': 0.0, 'bert_dropout_p': 0.1, 'ema_opt': 0, 'ema_gamma': 0.995, 'have_lr_scheduler': True, 'multi_step_lr': '10,20,30', 'freeze_layers': -1, 'embedding_opt': 0, 'lr_gamma': 0.5, 'bert_l2norm': 0.0, 'scheduler_type': 'ms', 'output_dir': 'checkpoints/scitail_tl_adamax_answer_opt0_gc0_ggc1_7_2_19', 'seed': 2018, 'task_config_path': 'configs/tasks_config.json', 'tasks_dropout_p': [0.1]}
    state_dict = torch.load(model_path)
    config = state_dict['config']
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    opt.update(config)
    model = MTDNNModel(opt, state_dict=state_dict, num_train_step=50)

    #actual prediction to be made: main outputs are predictions which is a list of size 1, and scores which is confidence in prediction for each class
    dev_metrics, dev_predictions, scores, golds, dev_ids = eval_model(
        model, dev_data, 0,use_cuda=True, with_label =False)
#model, data, metric_meta, use_cuda=True, with_label=True
    return dev_predictions, scores


preds, scores = make_prediction(Description_A, Description_B,
                                "checkpoint/sst_model_0.pt")
print(preds)
if preds[0] == 1:
    print("Matching")
else:
    print("Non-Matching")
