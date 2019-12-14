import argparse
import json
import os
import torch

from data_utils.task_def import TaskType
from experiments.exp_def import TaskDefs
from experiments.glue.glue_utils import eval_model
from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel

def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

parser = argparse.ArgumentParser()
parser.add_argument("--task_def", type=str, default="experiments/glue/glue_task_def.yml")
parser.add_argument("--task", type=str)
parser.add_argument("--task_id", type=int, help="the id of this task when training")

parser.add_argument("--prep_input", type=str)
parser.add_argument("--with_label", action="store_true")
parser.add_argument("--score", type=str, help="score output path")

parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--batch_size_eval', type=int, default=8)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')

parser.add_argument("--checkpoint", default='mt_dnn_models/bert_model_base_uncased.pt', type=str)

args = parser.parse_args()

# load task info
task_defs = TaskDefs(args.task_def)
assert args.task in task_defs.task_type_map
assert args.task in task_defs.data_type_map
assert args.task in task_defs.metric_meta_map
data_type = task_defs.data_type_map[args.task]
task_type = task_defs.task_type_map[args.task]
metric_meta = task_defs.metric_meta_map[args.task]
pw_task = False
if task_type == TaskType.Ranking:
    pw_task = True

# load data
test_data = BatchGen(BatchGen.load(args.prep_input, False, pairwise=pw_task, maxlen=args.max_seq_len),
                     batch_size=args.batch_size_eval,
                     gpu=args.cuda, is_train=False,
                     task_id=args.task_id,
                     maxlen=args.max_seq_len,
                     pairwise=pw_task,
                     data_type=data_type,
                     task_type=task_type)

# load model
checkpoint_path = args.checkpoint
assert os.path.exists(checkpoint_path)
if args.cuda:
    state_dict = torch.load(checkpoint_path)
else:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
config = state_dict['config']
config["cuda"] = args.cuda
# model = MTDNNModel(config, state_dict=state_dict)
opt = {'log_file': 'checkpoints/scitail_tl_adamax_answer_opt0_gc0_ggc1_7_2_19/log.log', 'init_checkpoint': '/data/kashyap_data/mt_dnn_models/mt_dnn_large_uncased.pt', 'data_dir': 'data/domain_adaptation', 'data_sort_on': False, 'name': 'farmer', 'train_datasets': ['snli'], 'test_datasets': ['snli'], 'pw_tasks': ['qnnli'], 'update_bert_opt': 0, 'multi_gpu_on': False, 'mem_cum_type': 'simple', 'answer_num_turn': 5, 'answer_mem_drop_p': 0.1, 'answer_att_hidden_size': 128, 'answer_att_type': 'bilinear', 'answer_rnn_type': 'gru', 'answer_sum_att_type': 'bilinear', 'answer_merge_opt': 1, 'answer_mem_type': 1, 'answer_dropout_p': 0.1, 'answer_weight_norm_on': False, 'dump_state_on': False, 'answer_opt': [
    0], 'label_size': '2', 'mtl_opt': 0, 'ratio': 0, 'mix_opt': 0, 'max_seq_len': 512, 'init_ratio': 1, 'cuda': True, 'log_per_updates': 500, 'epochs': 5, 'batch_size': 16, 'batch_size_eval': 8, 'optimizer': 'adamax', 'grad_clipping': 0.0, 'global_grad_clipping': 1.0, 'weight_decay': 0, 'learning_rate': 5e-05, 'momentum': 0, 'warmup': 0.1, 'warmup_schedule': 'warmup_linear', 'vb_dropout': True, 'dropout_p': 0.1, 'dropout_w': 0.0, 'bert_dropout_p': 0.1, 'ema_opt': 0, 'ema_gamma': 0.995, 'have_lr_scheduler': True, 'multi_step_lr': '10,20,30', 'freeze_layers': -1, 'embedding_opt': 0, 'lr_gamma': 0.5, 'bert_l2norm': 0.0, 'scheduler_type': 'ms', 'output_dir': 'checkpoints/scitail_tl_adamax_answer_opt0_gc0_ggc1_7_2_19', 'seed': 2018, 'task_config_path': 'configs/tasks_config.json', 'tasks_dropout_p': [0.1]}
state_dict = torch.load(
    "checkpoint/snli_model_0.pt")
config = state_dict['config']
config['attention_probs_dropout_prob'] = 0.1
config['hidden_dropout_prob'] = 0.1
opt.update(config)
model = MTDNNModel(opt, state_dict=state_dict, num_train_step=50)

test_metrics, test_predictions, scores, golds, test_ids = eval_model(model, test_data,
                                                                     metric_meta=metric_meta,
                                                                     use_cuda=args.cuda, with_label=args.with_label)

results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
dump(args.score, results)
if args.with_label:
    print(test_metrics)
