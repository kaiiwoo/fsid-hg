import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel

from DNNC_few_shot_intent.models.utils import get_train_dataloader, process_train_batch
from egnn.model import GraphNetwork

class HGraphNetwork(nn.Module):
    def __init__(self, args):
        """Hierarchical Graph Few-Shot Classifier: (Pre-trained) Roberta + (EGNN head x N)

        :param _type_ args: args
        """
        super(HGraphNetwork, self).__init__()
        self.args = args
        self.n_domain = args.n_domain
        
        self.backbone = None
        self.heads = nn.ModuleDict({f'head{i+1}' : GraphNetwork(args) for i in range(self.n_domain)})
        
    
    def initialize_node(self, batch):
        input_ids, input_mask, segment_ids, label_ids = batch #(batch_size, max_sent_len)
        
        config = AutoConfig.from_pretrained('roberta-base')
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        if args.path is not None:
            state_dict = torch.load(args.path + '/pytorch_model.bin')
            node_initializer = AutoModel.from_pretrained(args.path, state_dict=state_dict, config=config)
        else:
            node_initializer = AutoModel.from_pretrained('roberta-base', config=config)
            
        node_initializer.eval()

        features = []
        for ex in samples:
            tokens_a = tokenizer.tokenize(ex.text_a) #text_a = text, text_b = None (in train_fewshot.py)

            if len(tokens_a) > self.args.train.max_seq_length - 2: # [CLS] & [SEP] 토큰 넣을 자리 마련
                tokens_a = tokens_a[:(self.args.train.max_seq_length - 2)]

            # input format for transformer 
            # 그냥 self.toknizer(tokens_a)하면 다 나오는 것들아냐? 번거로워 보인다.
            tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
            segment_ids = [0] * len(tokens)
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            
            padding = [0] * (self.args.train.max_seq_length - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.args.train.max_seq_length
            assert len(input_mask) == self.args.train.max_seq_length
            assert len(segment_ids) == self.args.train.max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
            
        batch_size = int(self.args.train.batch_size / self.args.train.grad_acc_step)
        loader = get_train_dataloader(features, batch_size)
        return loader
    
    # cpft fine-tuing에 활용할 것
    def train(self, train_examples):
        """train/finetune for intent detection dataset for single trial

        Args:
            train_examples (list): [(sample 1, label 1), ..., (sample K, label 1), (sample 1, label 2), ...... ,(sample K, label L)]
            N * K samples for each trial
        """
        batch_size = int(self.args.train.batch_size / self.args.train.grad_acc_step)

        n_iter = int(len(train_examples) / batch_size / self.args.train.grad_acc_step * self.args.train.n_epoch)

        # AdamW -> follow configuration in DNNC paper
        optimizer, scheduler = get_optimizer(self.model, n_iter, self.args)
        
        # prepare trainloader
        train_features, label_distribution = self.convert_examples_to_features(train_examples, train = True)
        train_dataloader = get_train_dataloader(train_features, batch_size)
        
        # logger
        logger.info('***** Label distribution for label smoothing *****')
        logger.info(str(label_distribution))
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", self.args.train.batch_size)
        logger.info("  Num iters = %d", n_iter)

        # traininig loop
        self.model.zero_grad()
        self.model.train()
        for epoch in trange(int(self.args.train.n_epoch), desc="Epoch"):

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                
                processed_batch = process_train_batch(batch, self.device) # (train_batch_size, max_sent_length)->max_seq_len랑 다름
                self.initialize_node(processed_batch)
                
                input_ids, input_mask, segment_ids, label_ids = processed_batch

                outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids) 
                # Automodel for seq classification -> logits.shape[-1] = n_label
                logits = outputs[0] #(batch_size, n_label)
                loss = loss_with_label_smoothing(label_ids, logits, label_distribution, self.args.train.label_smoothing, self.device)
                
                self.writer.add_scalar("loss/train", loss.item(), epoch * len(train_dataloader) + (step + 1))
                
                # 배치사이즈가 너무 커서 GPU 용량 감당 못 할때 사용하는 기술
                # batchsize 512에 대해 BPP하고싶은데, 512못 쓸때 -> batchsize 128로 4번 돌릴때마다 각각 4번 나온 gradeint 다 합쳐서 한번 번에 업데이트 
                # batchsize 512에 대한 bpp 효과!!
                if self.args.train.grad_acc_step > 1:
                    loss = loss / self.args.train.grad_acc_step

                loss.backward()

                if (step + 1) % self.args.train.grad_acc_step == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.train.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()

            self.model.train()
    
    @torch.no_grad()
    def evaluate(self, eval_examples):
        """

        Args:
            eval_examples (list):

        Returns:
           list: [(pred_prob, pred_idx), ...] length=eval_batch_size
        """
        if len(eval_examples) == 0:
            logger.info('\n  No eval data!')
            return []

        eval_features = self.convert_examples_to_features(eval_examples, train = False)
        eval_dataloader = get_eval_dataloader(eval_features, self.args.eval.batch_size)

        self.model.eval()
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            logits = outputs[0] #(eval_batch_size, n_label)
            confs = torch.softmax(logits, dim=1)

            confs = confs.detach().cpu()

            for i in range(input_ids.size(0)):
                conf, index = confs[i].max(dim = 0)
                preds.append((conf.item(), self.label_list[index.item()]))
                
        return preds
    
    def forward(self, samples):  
        feature_loader = self.initialize_node(samples) 

        output = {}
        for i in range(self.n_domain):
            output[f'head{i+1}'] = self.heads[f'head{i+1}'](feature_loader)
            
        return output