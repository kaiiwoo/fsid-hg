import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer



class BertEncoder(nn.Module):
    def __init__(self):
        """Bert-kind sentence encoder for initializing node features

        Args:
            args (args): arguments
        """
        super(BertEncoder, self).__init__()
        # self.args = args
        
        # default: RoBERTa
        self.config = AutoConfig.from_pretrained('roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        # self.config = AutoConfig.from_pretrained(args.bert_type)
        # self.tokenizer = AutoTokenizer.from_pretrained(args.bert_type)

        # if args.ckpt_dir is not None:
        #     state_dict = torch.load(args.ckpt_dir + '/pytorch_model.bin')
        #     self.model = AutoModel.from_pretrained(args.path, state_dict=state_dict, config=config)
        # else:
        # self.model = AutoModel.from_pretrained('roberta-base', config=config)
        
    def tokenize(self, batch: list):
        outputs = []
        for ex in batch:
            ids = self.tokenize(ex)

            if len(ids) > self.args.train.max_seq_len:
                ids = ids[:self.args.train.max_seq_len - 2]

            ids = [self.tokenizer.cls_token] + input_ids + [self.tokenizer.sep_token]
            pad_len = self.args.train.max_seq_len - len(ids)
            
            ids = ids + [0] * pad_len
            att_mask = [1] * len(ids) + [0] * pad_len
            seg_ids = [0] * len(ids) + [0] * pad_len
            
        
    def forward(self, batch):
        
        return self.tokenizer(batch)['input_ids']
        # return self.tokenizer(batch)
        
        
b = BertEncoder()
print(b(['i love you', "i do love you"])) #[list of [input-]]