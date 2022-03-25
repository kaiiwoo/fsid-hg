"""
Fine-tune the model w/ processed till 1)self-supervised contrastvie learning
load path from SimCSE/runs

adapted from DNNC-few-shot intewnt 
"""


import argparse
import random
import logging
import os
from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm
from collections import defaultdict
from omegaconf import OmegaConf

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from DNNC_few_shot_intent.models.classifier import Classifier
from data_utils import load_intent_datasets, load_intent_examples, InputExample, sample


def set_random_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_acc(examples, in_domain_preds):
    gt = np.array([e.label for e in examples])
    preds = np.array([pred[1] for pred in in_domain_preds])
    acc= (gt == preds).sum() / preds.shape[0]
    return acc
    
    
def tabular_print(cache):
    result_dict = {'accuracy' : cache}
    print(tabulate(result_dict, headers='keys', tablefmt='github', showindex=False))


def train(args, writer):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    # filehandler ref: https://inma.tistory.com/136
    filehandler = logging.FileHandler(os.path.join(args.output_dir, 'result.log'))
    logger.addHandler(filehandler)
    
    set_random_seeds(args.seed)

    K = args.n_shot
    T = args.train.n_trial

    # load datasets
    train_path = os.path.join(args.data_dir, 'train')
    val_path = os.path.join(args.data_dir, 'valid')
    test_path = os.path.join(args.data_dir, 'test')
    train_set, val_set, test_set = load_intent_datasets(train_path, val_path, test_path, args.train.do_lower_case)

    # get trainset where each task(label) contains K samples. repeat for T times
    sampled_tasks = [sample(K, train_set) for _ in range(T)]

    label_lists = []
    intent_train_examples = []
    intent_val_examples = []
    intent_test_examples = []


    # Task batch 만들기
    for i in range(T):
        tasks = sampled_tasks[i]
        label_lists.append([])
        intent_train_examples.append([])
        #[ (1st trial)[(sample 1, label 1), (sample N_1, label 1), ....., (sample N_L, label L)], (2nd trial)[...] ] N_i: # of samples whose label is i
        intent_val_examples.append([InputExample(e.text, None, e.label) for e in val_set])
        intent_test_examples.append([InputExample(e.text, None, e.label) for e in test_set])

        for task in tasks: # # of tasks = # of all labels 
            label = task['task']
            examples = task['examples']
            #[ (1st trial)[label 1, label 2, ...., label L], (2nd trial)[...] ]
            label_lists[-1].append(label)

            for j in range(len(examples)):
                # [ (1st trial)[(sample 1, label 1), ..., (sample K, label 1), (sample 1, label 2), ...... ,(sample K, label L)],  (2nd trial)[...] ]
                # N * K samples for each trial
                intent_train_examples[-1].append(InputExample(examples[j], None, label))
            
        

    # 본격 Training loop
    val_cache = []
    test_cache = []
    best_trial = 0
    best_val_acc = -1.0
    best_param = None
    for j in tqdm(range(T), desc="trials for examining robustness"):
        if args.ckpt_dir is not None: # with pre-trained
            logger.info("Intent Detection w/ Pre-trained body")
            model = Classifier(path = args.ckpt_dir,
                               label_list = label_lists[j],
                               args = args,
                               writer=writer)

        else: # from scratch
            logger.info("Intent Detection from scratch")
            model = Classifier(path = None,
                               label_list = label_lists[j],
                               args = args,
                               writer=writer)
            
        # fine-tune intent detection
        model.train(intent_train_examples[j])

        # validation
        val_preds = model.evaluate(intent_val_examples[j])
        # SequentialSampler 덕분에 val_set과 바로 비교가능!
        val_acc = get_acc(val_set, val_preds) 
        val_cache.append(val_acc)
        
        if val_acc > best_val_acc:
            best_trial = j+1
            best_val_acc = val_acc
            best_param = model
            
        test_preds = best_param.evaluate(intent_test_examples[j])
        test_acc = get_acc(test_set, test_preds) 
        test_cache.append(test_acc)
        
    tabular_print(test_cache)

    logger.info(f"{tabulate({'val accuracy': val_cache}, headers='keys', tablefmt='github', showindex=False)}")
    logger.info(f"{tabulate({'test accuracy': test_cache}, headers='keys', tablefmt='github', showindex=False)}")
    logger.info(f"best_val_acc:{best_val_acc}\tbest_trial:{best_trial}")
    logger.info(f"avg_test_acc:{sum(test_cache) / len(test_cache):.4f}")
    best_param.save(os.path.join(args.output_dir, f"best_model_trial_at{best_trial}"))
    logger.info("Model saved & Training has done!")
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./finetune_template.yaml')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    # only specify dataname & n_shot
    parser.add_argument('--data_name', choices=['ATIS', 'BANKING77', 'BANKING77-OOS', 'CLINC-SD-OOS', 'CLINC150', 'HWU64', 'SNIPS'], required=True)
    parser.add_argument('--n_shot', type=int, default=10)
    args = parser.parse_args()

    opt = OmegaConf.load(args.config)

    # aggregate argparse to omegaconf
    opt.data_dir = os.path.join(opt.data_dir, args.data_name)
    opt.ckpt_dir = args.ckpt_dir
    opt.n_shot = args.n_shot
    
    if args.ckpt_dir is not None:
        exp_name = f"{args.data_name}_{opt.train.n_epoch}epoch_{opt.train.batch_size}batchsize_finetune_{opt.n_shot}shot_seed{opt.seed}"
    else:
        exp_name = f"{args.data_name}_{opt.train.n_epoch}epoch_{opt.train.batch_size}batchsize_scratch_{opt.n_shot}shot_seed{opt.seed}"
    
    opt.output_dir = os.path.join(opt.log_dir, str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '_' + exp_name)
    
    os.makedirs(opt.output_dir, exist_ok=True)
    writer = SummaryWriter(opt.output_dir)

    # train
    train(opt, writer)

