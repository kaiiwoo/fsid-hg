from os import supports_bytes_environ
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random

# class IntentDataset(Dataset):
#     def __init__(self, path):
#         super(IntentDataset, self).__init__()
#         self.root = path
#         # self.tokenizer = AutoTokenizer(args.bert_type)
#         self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

#         # self.do_lower_case = args.train.do_lower_case
#         self.do_lower_case = True

#         self.samples = []
        
#         self.label2idx = {}
#         self.domain2idx = {}
        
#         self.get_gt_info(path)
#         self.cache_samples()
        
        
#     def get_gt_info(self, path):
#         with open(f"{path}/label", 'r', encoding='utf-8') as f_label:
#             i = 0
#             j = 0
            
#             for label in f_label:
#                 domain = label.split('_')[0]

#                 if label not in self.label2idx:
#                     self.label2idx[label] = i
#                     i += 1
                
#                 if domain not in self.domain2idx:
#                     self.domain2idx[domain] = j
#                     j += 1
                

#     def cache_samples(self):

#         with open('{}/seq.in'.format(self.root), 'r', encoding="utf-8") as f_text, \
#                 open('{}/label'.format(self.root), 'r', encoding="utf-8") as f_label:

#             for text, label in zip(f_text, f_label):
#                 if self.do_lower_case: 
#                     text = text.lower()
                    
#                 domain = label.split('_')[0]
                
#                 input_ids = self.tokenizer(text.lower())['input_ids']

#                 if len(input_ids) > 128 - 2:
#                     input_ids = input_ids[:128 - 2]
                    
#                 input_ids = [self.tokenizer.cls_token] + input_ids + [self.tokenizer.sep_token]
#                 att_mask = [1] * len(input_ids)
#                 seg_ids = [0] * len(input_ids)
                
#                 sample = {
#                     'input_ids': input_ids,
#                     'att_mask': att_mask,
#                     'seg_ids': seg_ids,
#                     'seq_len': len(input_ids),
#                     'label': self.label2idx[label],
#                     'domain' : self.domain2idx[domain]
#                 }
                
#                 self.samples.append(sample)
                
                
    
#     def get_task_batch(self,
#                        num_tasks=5,
#                        num_ways=20,
#                        num_shots=1,
#                        num_queries=1,
#                        seed=None):
        
#         if seed is not None:
#             random.seed(seed)

#         support_data, support_label, query_data, query_label = [], [], [], []
            
#         for t_idx in range(num_tasks):
#             task_classes = random.sample(self.label2idx.keys(), num_ways)

#             for c_idx in range(num_ways):
#                 target_data = [x for x in self.samples if x['label'] == task_classes[c_idx]]
#                 class_data_list = random.sample(target_data, num_shots + num_queries)

#                 for i_idx in range(num_shots):
#                     support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
#                     support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

#                 # load sample for query set
#                 for i_idx in range(num_queries):
#                     query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
#                     query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

#         support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
#         support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
#         query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
#         query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

#         return [support_data, support_label, query_data, query_label]

#     def __getitem__(self, idx):
#         return self.samples[idx]
                
#     def __len__(self):
#         return len(self.samples)

# def make_episode_batch(data):
#     max_len = 
    


# def CustomIntentLoader()

# # 전체 corpus에 대해 label / domain 정보를 만들어야 될 것 같은데. test / val / train에 적용하면 서로 딕셔너리 다르게 나올텐데.
# # 내가 가진 datasets은 이미 전처리 된 거라서 train / val/에 대해 똑같은 룩업 딕셔너리 생성됐음..
# d = IntentDataset('/home/keonwookim/something-FSID/datasets/HWU64/test')
# print(d.label2idx)
# print(d.domain2idx)
# print(len(d))

# 8954 + 1076
class InputExample(object):

    def __init__(self, text_a, text_b, label = None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
class IntentExample:
    def __init__(self, text, label, do_lower_case):
        self.original_text = text
        self.text = text
        self.label = label

        if do_lower_case:
            self.text = self.text.lower()
            

def load_intent_examples(file_path, do_lower_case):
    examples = []

    with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            # [(text, label) ....]
            e = IntentExample(text.strip(), label.strip(), do_lower_case)
            examples.append(e)

    return examples

def load_intent_datasets(train_file_path, dev_file_path, test_file_path, do_lower_case):
    train_examples = load_intent_examples(train_file_path, do_lower_case)
    dev_examples = load_intent_examples(dev_file_path, do_lower_case)
    test_examples = load_intent_examples(test_file_path, do_lower_case)

    return train_examples, dev_examples, test_examples

def sample(K, examples):
    labels = {} # unique classes

    for e in examples:
        if e.label in labels:
            labels[e.label].append(e.text)
        else:
            labels[e.label] = [e.text]

    # 그냥 한 레이블 내 데이터샘플들 셔플하고 k개 뽑기
    # len(sampled_exampled) = L (total # of labels)
    sampled_examples = []
    for l in labels:
        random.shuffle(labels[l])
        if l == 'oos':
            examples = labels[l][:K]
        else:
            examples = labels[l][:K]
        sampled_examples.append({'task': l, 'examples': examples})
    return sampled_examples