import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from tokenizer import BertTokenizer
from optimizer import AdamW
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
from tqdm import tqdm
from itertools import cycle
from pcgrad import PCGrad
from pcgrad_amp import PCGradAMP
from gradvac_amp import GradVacAMP
import copy

from smart_regularization import smart_regularization

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_multitask, test_model_multitask, \
    model_eval_paraphrase, model_eval_sts, model_eval_sentiment


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


TQDM_DISABLE = False

class Colors:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
N_STS_CLASSES = 6

def get_term_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'finetune':
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Step 2: Add a linear layer for sentiment classification
        self.dropout_sentiment = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_sentiment = nn.ModuleList([nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)])

        # Step 3: Add a linear layer for paraphrase detection
        self.dropout_paraphrase = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_paraphrase = nn.ModuleList([nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [nn.Linear(BERT_HIDDEN_SIZE, 1)])

        # Step 4: Add a linear layer for semantic textual similarity
        # This is a regression task, so the output should be a single number
        self.dropout_similarity = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_similarity = nn.ModuleList([nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [nn.Linear(BERT_HIDDEN_SIZE, 1)])


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        
        # Step 1: Get the BERT embeddings
        bert_output = self.bert(input_ids, attention_mask)

        # Step 2: Get the [CLS] token embeddings
        cls_embeddings = bert_output['pooler_output']
        return cls_embeddings

    def last_layers_sentiment(self, x):
        """Given a batch of sentences embeddings, outputs logits for classifying sentiment."""
        # Step 2: Hidden layers
        for i in range(len(self.linear_sentiment) - 1):
            x = self.dropout_sentiment[i](x)
            x = self.linear_sentiment[i](x)
            x = F.relu(x)

        # Step 3: Final layer
        x = self.dropout_sentiment[-1](x)
        logits = self.linear_sentiment[-1](x)
        # logits = F.softmax(logits, dim=1)
        return logits
    
    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        # Step 1: Get the BERT embeddings
        x = self.forward(input_ids, attention_mask)
        return self.last_layers_sentiment(x)

    def get_similarity_paraphrase_embeddings(self, input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, get the BERT embeddings.'''
        # Step 0: Get [SEP] token ids
        sep_token_id = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)

        # Step 1: Concatenate the two sentences in: sent1 [SEP] sent2 [SEP]
        input_id = torch.cat((input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(batch_sep_token_id), attention_mask_2, torch.ones_like(batch_sep_token_id)), dim=1)

        # Step 2: Get the BERT embeddings
        x = self.forward(input_id, attention_mask)

        return x
    
    def last_layers_paraphrase(self, x):
        """Given a batch of pairs of sentences embedding, outputs logits for predicting whether they are paraphrases."""
        #Step 2: Hidden layers
        for i in range(len(self.linear_paraphrase) - 1):
            x = self.dropout_paraphrase[i](x)
            x = self.linear_paraphrase[i](x)
            x = F.relu(x)

        # Step 3: Final layer
        x = self.dropout_paraphrase[-1](x)
        logits = self.linear_paraphrase[-1](x)
        # logits = torch.sigmoid(logits)
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # Step 1: Get the BERT embeddings
        x = self.get_similarity_paraphrase_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        return self.last_layers_paraphrase(x)


    def last_layers_similarity(self, x):
        """Given a batch of pairs of sentences embeddings, outputs logits for predicting how similar they are."""
        # Step 3: Hidden layers
        for i in range(len(self.linear_similarity) - 1):
            x = self.dropout_similarity[i](x)
            x = self.linear_similarity[i](x)
            x = F.relu(x)

        # Step 4: Final layer
        x = self.dropout_similarity[-1](x)
        preds = self.linear_similarity[-1](x)
        # preds = torch.sigmoid(preds) * 6 - 0.5 # Scale to [-0.5, 5.5]

        # # If we are evaluating, then we cap the predictions to the range [0, 5]
        # if not self.training:
        #     preds = torch.clamp(preds, 0, 5)
        return preds
    
    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # Step 1 : Get the BERT embeddings
        x = self.get_similarity_paraphrase_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        return self.last_layers_similarity(x)


class ObjectsGroup:

    def __init__(self, model, optimizer, scaler = None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.loss_sum = 0

class Scheduler:

    def __init__(self, dataloaders, reset=True):
        self.dataloaders = dataloaders
        self.names = list(dataloaders.keys())
        if reset: self.reset()

    def reset(self):
        self.sst_iter = iter(self.dataloaders['sst'])
        self.para_iter = iter(self.dataloaders['para'])
        self.sts_iter = iter(self.dataloaders['sts'])
        self.steps = {'sst': 0, 'para': 0, 'sts': 0}

    def get_SST_batch(self):
        try:
            return next(self.sst_iter)
        except StopIteration:
            self.sst_iter = cycle(self.dataloaders['sst'])
            return next(self.sst_iter)

    def get_Paraphrase_batch(self):
        try:
            return next(self.para_iter)
        except StopIteration:
            self.para_iter = cycle(self.dataloaders['para'])
            return next(self.para_iter)

    def get_STS_batch(self):
        try:
            return next(self.sts_iter)
        except StopIteration:
            self.sts_iter = cycle(self.dataloaders['sts'])
            return next(self.sts_iter)

    def get_batch(self, name: str):
        if name == "sst": return self.get_SST_batch()
        elif name == "para": return self.get_Paraphrase_batch()
        elif name == "sts": return self.get_STS_batch()
        raise ValueError(f"Unknown batch name: {name}")

    def process_named_batch(self, objects_group: ObjectsGroup, args: dict, name: str, apply_optimization: bool = True):
        batch = self.get_batch(name)
        process_fn, gradient_accumulations = None, 0
        if name == "sst":
            process_fn = process_sentiment_batch
            gradient_accumulations = args.gradient_accumulations_sst
        elif name == "para":
            process_fn = process_paraphrase_batch
            gradient_accumulations = args.gradient_accumulations_para
        elif name == "sts":
            process_fn = process_similarity_batch
            gradient_accumulations = args.gradient_accumulations_sts
        else:
            raise ValueError(f"Unknown batch name: {name}")
        
        # Process the batch
        loss_of_batch = 0
        for _ in range(gradient_accumulations):
            loss_of_batch += process_fn(batch, objects_group, args)

        # Update the model
        self.steps[name] += 1
        if apply_optimization: step_optimizer(objects_group, args, step=self.steps[name])

        return loss_of_batch


class RandomScheduler(Scheduler):

    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset=True)

    def process_one_batch(self, epoch: int, num_epochs: int, objects_group: ObjectsGroup, args: dict):
        name = random.choice(self.names)
        return name, self.process_named_batch(objects_group, args, name)


class RoundRobinScheduler(Scheduler):

    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset=False)
        self.reset()

    def reset(self):
        self.index = 0
        return super().reset()

    def process_one_batch(self, epoch: int, num_epochs: int, objects_group: ObjectsGroup, args: dict):
        name = self.names[self.index]
        self.index = (self.index + 1) % len(self.names)
        return name, self.process_named_batch(objects_group, args, name)


class PalScheduler(Scheduler):

    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset=False)
        self.sizes = np.array([len(dataloaders[dataset]) for dataset in self.names])
        self.reset()

    def process_one_batch(self, epoch: int, num_epochs: int, objects_group: ObjectsGroup, args: dict):
        alpha = 0.2
        if num_epochs > 1: alpha = 1 - 0.8 * (epoch - 1) / (num_epochs - 1) 
        probs = self.sizes ** alpha
        probs /= np.sum(probs)

        # Sample a dataset
        name = np.random.choice(self.names, p=probs)
        return name, self.process_named_batch(objects_group, args, name)


def process_sentiment_batch(batch, objects_group: ObjectsGroup, args: dict):
    device = args.device
    model, scaler = objects_group.model, objects_group.scaler

    with autocast() if args.use_amp else nullcontext():
        b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
        b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

        embeddings = model.forward(b_ids, b_mask)
        logits = model.last_layers_sentiment(embeddings)
        
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
        loss_value = loss.item()
        
        if args.use_smart_regularization:
            smart_regularization(loss_value, args.smart_weight_regularization, embeddings, logits, model.last_layers_sentiment)

        objects_group.loss_sum += loss_value

        if args.projection == "none":
            if args.use_amp: scaler.scale(loss).backward()
            else: loss.backward()
        return loss


def process_paraphrase_batch(batch, objects_group: ObjectsGroup, args: dict):
    device = args.device
    model, scaler = objects_group.model, objects_group.scaler

    with autocast() if args.use_amp else nullcontext():
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)

        embeddings = model.get_similarity_paraphrase_embeddings(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        preds = model.last_layers_paraphrase(embeddings)
        loss = F.binary_cross_entropy_with_logits(preds.view(-1), b_labels.float(), reduction='sum') / args.batch_size
        loss_value = loss.item()

        if args.use_smart_regularization:
            smart_regularization(loss_value, args.smart_weight_regularization, embeddings, preds, model.last_layers_paraphrase)

        objects_group.loss_sum += loss_value
        
        if args.projection == "none":
            if args.use_amp: scaler.scale(loss).backward()
            else: loss.backward()
        return loss


def process_similarity_batch(batch, objects_group: ObjectsGroup, args: dict):
    device = args.device
    model, scaler = objects_group.model, objects_group.scaler

    with autocast() if args.use_amp else nullcontext():
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)

        embeddings = model.get_similarity_paraphrase_embeddings(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        preds = model.last_layers_similarity(embeddings)
        loss = F.mse_loss(preds.view(-1), b_labels.view(-1), reduction='sum') / args.batch_size
        loss_value = loss.item()

        if args.use_smart_regularization:
            smart_regularization(loss_value, args.smart_weight_regularization, embeddings, preds, model.last_layers_similarity)

        objects_group.loss_sum += loss_value
        
        if args.projection == "none":
            if args.use_amp: scaler.scale(loss).backward()
            else: loss.backward()
        return loss

def step_optimizer(objects_group: ObjectsGroup, args: dict, step: int, total_nb_batches = None):
    """Step the optimizer and update the scaler. Returns the loss"""
    optimizer, scaler = objects_group.optimizer, objects_group.scaler
    if args.use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()
    loss_value = objects_group.loss_sum
    objects_group.loss_sum = 0
    torch.cuda.empty_cache()
    if TQDM_DISABLE:
        str_total_nb_batches = "?" if total_nb_batches is None else str(total_nb_batches)
        print(f'batch {step}/{str_total_nb_batches} STS - loss: {loss_value:.3f}')
    return loss_value

def finish_training_batch(objects_group: ObjectsGroup, args: dict, step: int, gradient_accumulations: int, total_nb_batches = None):
    """Finish training a batch and return whether the model is updated"""
    if step % gradient_accumulations == 0:
        step_optimizer(objects_group, args, step, total_nb_batches)
        return True
    return False


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    # print(f"save the model to {filepath}")
    return filepath


def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')
    print("")

    # SST: Sentiment classification
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size_sst,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size_sst,
                                    collate_fn=sst_dev_data.collate_fn)

    # Para: Paraphrase detection
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size_para,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size_para,
                                    collate_fn=para_dev_data.collate_fn)

    # STS: Semantic textual similarity
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size_sts,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size_sts,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'pretrained_model_name': args.pretrained_model_name,
              'n_hidden_layers': args.n_hidden_layers}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    if args.pretrained_model_name != "none":
        config = load_model(model, args.pretrained_model_name)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = None if not args.use_amp else GradScaler()

    if args.projection == 'pcgrad':
        optimizer = PCGrad(optimizer) if not args.use_amp else PCGradAMP(num_tasks=3, optimizer=optimizer, scaler=scaler)
    elif args.projection == 'vaccine':
        optimizer = GradVacAMP(num_tasks=3, optimizer=optimizer, scaler=scaler, DEVICE=device, beta=args.beta_vaccine)
    best_dev_acc = 0
    best_dev_accuracies = {'sst': 0, 'para': 0, 'sts': 0}
    best_dev_rel_improv = 0

    # Package objects
    objects_group = ObjectsGroup(model, optimizer, scaler)
    args.device = device
    dataloaders = {'sst': sst_train_dataloader, 'para': para_train_dataloader, 'sts': sts_train_dataloader}
    scheduler = None
    if args.task_scheduler == 'round_robin':
        scheduler = RoundRobinScheduler(dataloaders)
    elif args.task_scheduler == 'pal':
        scheduler = PalScheduler(dataloaders)
    elif args.task_scheduler == 'random':
        scheduler = RandomScheduler(dataloaders)


    # Loss logs
    train_loss_logs_epochs = {'sst': [], 'para': [], 'sts': []}
    dev_acc_logs_epochs = {'sst': [], 'para': [], 'sts': []}

    # ==================== THIS IS INDIVIDUAL PRETRAINING ====================

    # Since we are pretraining, we are only updating the layers on top off BERT
    # This means that the tasks are not dependent on each other
    # We can therefore train them in parallel ans save the best state for each task
    # At the end, we load the best state for each task and evaluate the model on the dev set (multitask)

    if args.option == 'individual_pretrain':
        # Dict to train each task separately
        infos = {'sst': {'num_batches': len(sst_train_dataloader), 'eval_fn': model_eval_sentiment, 'dev_dataloader': sst_dev_dataloader, 'best_dev_acc': 0, 'best_model': None, 'layer': model.linear_sentiment},
                'para': {'num_batches': len(para_train_dataloader), 'eval_fn': model_eval_paraphrase, 'dev_dataloader': para_dev_dataloader, 'best_dev_acc': 0, 'best_model': None, 'layer': model.linear_paraphrase},
                'sts':  {'num_batches': len(sts_train_dataloader), 'eval_fn': model_eval_sts, 'dev_dataloader': sts_dev_dataloader, 'best_dev_acc': 0, 'best_model': None, 'layer': model.linear_similarity}}
        
        for task in ['sst', 'sts', 'para']:
            optimizer = AdamW(model.parameters(), lr=lr)
            terminal_width = get_term_width()
            last_improv = -1
            print(Colors.BOLD + f'{"     Individually Pretraining " + task + "     ":-^{get_term_width()}}' + Colors.END)
            for epoch in range(args.epochs):
                for i in tqdm(range(infos[task]['num_batches']), desc=task + ' epoch ' + str(epoch), disable=TQDM_DISABLE, smoothing=0):
                    loss = scheduler.process_named_batch(name=task, objects_group=objects_group, args=args)
                    if not args.no_tensorboard: writer.add_scalar("Loss pretrain " + task, loss.item(), args.batch_size * (epoch * infos[task]['num_batches'] + i))
                
                # Evaluate on dev set
                color_score, saved = Colors.BLUE, False
                dev_acc, _, _, _ = infos[task]['eval_fn'](infos[task]['dev_dataloader'], model, device)
                if dev_acc > infos[task]['best_dev_acc']:
                    infos[task]['best_dev_acc'] = dev_acc
                    infos[task]['best_model'] = copy.deepcopy(infos[task]['layer'].state_dict())
                    color_score, saved = Colors.PURPLE, True
                    last_improv = epoch
                if not args.no_tensorboard: writer.add_scalar("[EPOCH] Dev accuracy " + task, dev_acc, epoch)
                if not args.no_tensorboard: writer.add_scalar("Dev accuracy " + task, dev_acc, epoch * args.batch_size_sts * infos[task]['num_batches'])
                
                # Print dev accuracy
                spaces_per_task = int((terminal_width - 3*(20+5)) / 2)
                end_print = f'{"Saved":>{25 + spaces_per_task}}' if saved else ""
                print(Colors.BOLD + color_score + f'{"Cur acc dev: ":<20}'   + Colors.END + color_score + f"{dev_acc:.3f}" + " " * spaces_per_task
                    + Colors.BOLD + color_score + f'{" Best acc dev: ":<20}' + Colors.END + color_score + f"{infos[task]['best_dev_acc']:.3f}"
                    + end_print + Colors.END)

                if epoch != args.epochs - 1: print("")
                elif epoch - last_improv >= args.patience:
                    print(Colors.BOLD + Colors.RED + f'{"Early stopping":^{get_term_width()}}' + Colors.END)
                    break
            print("-" * terminal_width)
            print('\n\n')

        # Load best model for each task
        for task in infos.keys():
            infos[task]['layer'].load_state_dict(infos[task]['best_model'])
        
        # Evaluate on dev set
        print(Colors.BOLD + Colors.CYAN + f'{"     Evaluation Multitask     ":-^{get_term_width()}}' + Colors.END + Colors.CYAN)
        (paraphrase_accuracy, _, _,
         sentiment_accuracy, _, _,
         sts_corr, _, _) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device, writer=writer, epoch=0, tensorboard=not args.no_tensorboard)
        print(Colors.BOLD + Colors.CYAN + f'{"Dev acc SST: ":<20}'    + Colors.END + Colors.CYAN + f"{sentiment_accuracy:.3f}" + " " * spaces_per_task
            + Colors.BOLD + Colors.CYAN + f'{" Dev acc Para: ":<20}'  + Colors.END + Colors.CYAN + f"{paraphrase_accuracy:.3f}" + " " * spaces_per_task
            + Colors.BOLD + Colors.CYAN + f'{" Dev acc STS: ":<20}'   + Colors.END + Colors.CYAN + f"{sts_corr:.3f}")

        # Save model
        saved_path = save_model(model, optimizer, args, config, args.filepath)
        print(Colors.BOLD + "Saved model to: ", saved_path + Colors.END + Colors.CYAN)
        print("-" * terminal_width + Colors.END)
        print("")
        return



    # ====================== THIS IS FINETUNING ======================

    # Run for the specified number of epochs
    # Here we don't even specify explicitly to reset the scheduler at the end of each epoch (i.e. reset the dataloaders).
    # This way we make sure that the scheduler goes through the entire dataset before resetting.
    # The num_of_batches is simply defined to be consistent with the size of the datasets.

    num_batches_per_epoch = args.num_batches_per_epoch
    if num_batches_per_epoch <= 0:
        num_batches_per_epoch = int(len(sst_train_dataloader) / args.gradient_accumulations_sst) + \
                                int(len(para_train_dataloader) / args.gradient_accumulations_para) + \
                                int(len(sts_train_dataloader) / args.gradient_accumulations_sts)
    
    last_improv = -1
    for epoch in range(args.epochs):
        print(Colors.BOLD + f'{"     Epoch " + str(epoch) + "     ":-^{get_term_width()}}' + Colors.END)
        model.train()
        train_loss = {'sst': 0, 'para': 0, 'sts': 0}
        num_batches = {'sst': 0, 'para': 0, 'sts': 0}

        if args.projection != "none":
            for i in tqdm(range(int(num_batches_per_epoch / 3)), desc=f'Train {epoch}', disable=TQDM_DISABLE, smoothing=0):
                losses = []
                for j, name in enumerate(['sst', 'sts', 'para']):
                    losses.append(scheduler.process_named_batch(objects_group=objects_group, args=args, name=name, apply_optimization=False))
                    train_loss[name] += losses[-1].item()
                    if not args.no_tensorboard: writer.add_scalar("Loss " + args.option + " " + name, losses[-1].item(), args.batch_size * (epoch * num_batches_per_epoch + 3 * i + j))
                    num_batches[name] += 1
                optimizer.backward(losses)
                optimizer.step()
        else:
            for i in tqdm(range(num_batches_per_epoch), desc=f'Train {epoch}', disable=TQDM_DISABLE, smoothing=0):
                task, loss = scheduler.process_one_batch(epoch=epoch+1, num_epochs=args.epochs, objects_group=objects_group, args=args)
                train_loss[task] += loss.item()
                if not args.no_tensorboard: writer.add_scalar("Loss " + args.option + " " + task, loss.item(), args.batch_size * (epoch * num_batches_per_epoch + i))
                num_batches[task] += 1

        # Compute average train loss
        for task in train_loss:
            train_loss[task] = 0 if num_batches[task] == 0 else train_loss[task] / num_batches[task]
            train_loss_logs_epochs[task].append(train_loss[task])

        # Eval on dev
        (paraphrase_accuracy, _, _,
        sentiment_accuracy,_, _,
        sts_corr, _, _) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device, writer=writer, epoch=epoch, tensorboard=not args.no_tensorboard)
        
        #We keep track of the accuracies for each task for each epoch
        dev_acc_logs_epochs['sst'].append(sentiment_accuracy)
        dev_acc_logs_epochs['para'].append(paraphrase_accuracy)
        dev_acc_logs_epochs['sts'].append(sts_corr)

        # Useful for deg
        # paraphrase_accuracy, sentiment_accuracy, sts_corr = 0.6, 0.4, 0.33333333

        # Computes relative improvement compared to a random baseline and to the best model so far
        # So 0, corresponds to a random baseline and 1 to the best model so far
        random_accuracies = {'sst': 1./N_SENTIMENT_CLASSES, 'para': 0.5, 'sts': 0.}
        best_accuracies_so_far = {'sst': 0.598, 'para': 0.924, 'sts': 0.929} # source: https://paperswithcode.com
        para_rel_improvement = (paraphrase_accuracy - random_accuracies['para']) / (best_accuracies_so_far['para'] - random_accuracies['para'])
        sst_rel_improvement = (sentiment_accuracy - random_accuracies['sst']) / (best_accuracies_so_far['sst'] - random_accuracies['sst'])
        sts_rel_improvement = (sts_corr - random_accuracies['sts']) / (best_accuracies_so_far['sts'] - random_accuracies['sts'])
        geom_mean_rel_improvement = (para_rel_improvement * sst_rel_improvement * sts_rel_improvement) ** (1/3)

        # Computes arithmetic average of the accuracies (used for the leaderboard)
        arithmetic_mean_acc = (paraphrase_accuracy + sentiment_accuracy + sts_corr) / 3
        
        # Write to tensorboard
        if not args.no_tensorboard: 
            writer.add_scalar("[EPOCH] Dev accuracy sst", sentiment_accuracy, epoch)
            writer.add_scalar("[EPOCH] Dev accuracy para", paraphrase_accuracy, epoch)
            writer.add_scalar("[EPOCH] Dev accuracy sts", sts_corr, epoch)
            writer.add_scalar("[EPOCH] Dev accuracy mean", arithmetic_mean_acc, epoch)
            writer.add_scalar("[EPOCH] Num batches sst", num_batches['sst'], epoch)
            writer.add_scalar("[EPOCH] Num batches para", num_batches['para'], epoch)
            writer.add_scalar("[EPOCH] Num batches sts", num_batches['sts'], epoch)
            writer.add_scalar("Dev accuracy sst", sentiment_accuracy, epoch * args.batch_size * num_batches_per_epoch)
            writer.add_scalar("Dev accuracy para", paraphrase_accuracy, epoch * args.batch_size * num_batches_per_epoch)
            writer.add_scalar("Dev accuracy sts", sts_corr, epoch * args.batch_size * num_batches_per_epoch)
            writer.add_scalar("Dev accuracy mean", arithmetic_mean_acc, epoch * args.batch_size * num_batches_per_epoch)
            writer.add_scalar("Num batches sst", num_batches['sst'], epoch * args.batch_size * num_batches_per_epoch)
            writer.add_scalar("Num batches para", num_batches['para'], epoch * args.batch_size * num_batches_per_epoch)
            writer.add_scalar("Num batches sts", num_batches['sts'], epoch * args.batch_size * num_batches_per_epoch)

        # Saves model if it is the best one so far on the dev set
        color_score, saved = Colors.BLUE, False
        if arithmetic_mean_acc > best_dev_acc:
            best_dev_acc = arithmetic_mean_acc
            best_dev_rel_improv = geom_mean_rel_improvement
            best_dev_accuracies = {'sst': sentiment_accuracy, 'para': paraphrase_accuracy, 'sts': sts_corr}
            saved_path = save_model(model, optimizer, args, config, args.filepath)
            color_score, saved = Colors.PURPLE, True
            last_improv = epoch

        terminal_width = get_term_width()
        spaces_per_task = int((terminal_width - 3*(20+5)) / 2)
        print(Colors.BOLD + f'{"Num batches SST: ":<20}'   + Colors.END + f"{num_batches['sst']:<5}" + " " * spaces_per_task
            + Colors.BOLD + f'{" Num batches Para: ":<20}' + Colors.END + f"{num_batches['para']:<5}" + " " * spaces_per_task
            + Colors.BOLD + f'{" Num batches STS: ":<20}'  + Colors.END + f"{num_batches['sts']:<5}")
        print(Colors.BOLD + f'{"Train loss SST: ":<20}'   + Colors.END  + f"{train_loss['sst']:.3f}" + " " * spaces_per_task
            + Colors.BOLD + f'{" Train loss Para: ":<20}' + Colors.END  + f"{train_loss['para']:.3f}" + " " * spaces_per_task
            + Colors.BOLD + f'{" Train loss STS: ":<20}'  + Colors.END  + f"{train_loss['sts']:.3f}")
        print(Colors.BOLD + Colors.CYAN + f'{"Dev acc SST: ":<20}'    + Colors.END + Colors.CYAN + f"{sentiment_accuracy:.3f}" + " " * spaces_per_task
            + Colors.BOLD + Colors.CYAN + f'{" Dev acc Para: ":<20}'  + Colors.END + Colors.CYAN + f"{paraphrase_accuracy:.3f}" + " " * spaces_per_task
            + Colors.BOLD + Colors.CYAN + f'{" Dev acc STS: ":<20}'   + Colors.END + Colors.CYAN + f"{sts_corr:.3f}")
        print(Colors.BOLD + color_score + f'{"Best acc SST: ":<20}'   + Colors.END + color_score + f"{best_dev_accuracies['sst']:.3f}" + " " * spaces_per_task
            + Colors.BOLD + color_score + f'{" Best acc Para: ":<20}' + Colors.END + color_score + f"{best_dev_accuracies['para']:.3f}" + " " * spaces_per_task
            + Colors.BOLD + color_score + f'{" Best acc STS: ":<20}'  + Colors.END + color_score + f"{best_dev_accuracies['sts']:.3f}")
        end_print = f'{"Saved to: " + saved_path:>{25 + spaces_per_task}}' if saved else ""
        print(Colors.BOLD + color_score + f'{"Mean acc dev: ":<20}'   + Colors.END + color_score + f"{arithmetic_mean_acc:.3f}" + " " * spaces_per_task
            + Colors.BOLD + color_score + f'{" Best mean acc: ":<20}' + Colors.END + color_score + f"{best_dev_acc:.3f}"
            + end_print + Colors.END)
        print(Colors.BOLD + f'{"Rel improv dev: ":<20}'   + Colors.END + f"{geom_mean_rel_improvement:.3f}" + " " * spaces_per_task
            + Colors.BOLD + f'{" Best rel improv: ":<20}' + Colors.END + f"{best_dev_rel_improv:.3f}")
        print("-" * terminal_width)
        print("")

        if epoch - last_improv >= args.patience:
            print(Colors.BOLD + Colors.RED + f'{"Early stopping":^{get_term_width()}}' + Colors.END)
            break
    
    if args.save_loss_acc_logs:
        # Write train_loss_logs_epochs to file
        with open(args.log_dir + '/train_loss.txt', 'w') as f:
            # Loop through the dictionary items and write them to the file
            for key, value in train_loss_logs_epochs.items():
                f.write('{}: {}\n'.format(key, value))
        #Write dev_acc_logs_epochs to file
        with open(args.log_dir + '/dev_acc.txt', 'w') as f:
            # Loop through the dictionary items and write them to the file
            for key, value in dev_acc_logs_epochs.items():
                f.write('{}: {}\n'.format(key, value))


def load_model(model, filepath):
    with torch.no_grad():
        saved = torch.load(filepath)
        config = saved['model_config']
        model.load_state_dict(saved['model'])
        print(f"Loaded model from {filepath}")
        return config

def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def print_subset_of_args(args, title, list_of_args, color = Colors.BLUE, print_length = 50, var_length = 15):
    """Prints a subset of the arguments in a nice format."""
    print("\n" + color + f'{" " + title + " ":█^{print_length}}')
    for arg in list_of_args:
        print(Colors.BOLD + f'█ {arg + ": ": >{var_length}}' + Colors.END + f'{getattr(args, arg): <{print_length - var_length - 3}}' +  color  + '█')
    print("█" * print_length + Colors.END)

def warn(message: str, color: str = Colors.RED) -> None:
    print(color + "WARNING: " + message + Colors.END)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--no_tensorboard", action='store_true', help="Dont log to tensorboard")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune', 'test', 'individual_pretrain'), default="pretrain")
    parser.add_argument("--pretrained_model_name", type=str, default="none")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="/predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="/predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="/predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="/predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="/predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="/predictions/sts-test-output.csv")

    #Arugment to save logs through the epochs (train loss and dev accuracy)
    parser.add_argument("--save_loss_acc_logs", type=bool, default=False)

    # hyper parameters
    parser.add_argument("--batch_size", help='This is the simulated batch size using gradient accumulations', type=int, default=256)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--n_hidden_layers", type=int, default=2, help="Number of hidden layers for the classifier")
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--num_batches_per_epoch", type=int, default=-1)
    parser.add_argument("--task_scheduler", type=str, choices=('random', 'round_robin', 'pal'), default="round_robin")

    # Optimizations
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--max_batch_size_sst", type=int, default=64)
    parser.add_argument("--max_batch_size_para", type=int, default=32)
    parser.add_argument("--max_batch_size_sts", type=int, default=64)
    parser.add_argument("--projection", type=str, choices=('none', 'pcgrad', 'vaccine'), default="none")
    parser.add_argument("--beta_vaccine", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, help="Number maximum of epochs without improvement", default=5)
    parser.add_argument("--use_smart_regularization", action='store_true')
    parser.add_argument("--smart_weight_regularization", type=float, default=1e-2)

    args = parser.parse_args()

    # Logs the command to recreate the same run with all the arguments
    s = "python3 multitask_classifier.py"
    for arg in vars(args):
        value = getattr(args, arg)
        if type(value) == bool:
            if value:
                s += f" --{arg}"
        else:
            s += f" --{arg} {value}"
    print("\n" + Colors.BOLD + "Command to recreate this run:" + Colors.END + s + "\n")

    # Saves s in args.log_dir/command.txt
    args.log_dir = writer.log_dir # Get the path of the folder where TensorBoard logs will be saved
    with open(os.path.join(args.log_dir, "command.txt"), "w") as f:
        f.write(s)

    # Makes sure that the actual batch sizes are not too large
    # Gradient accumulations are used to simulate larger batch sizes
    args.gradient_accumulations_sst = int(np.ceil(args.batch_size / args.max_batch_size_sst))
    args.batch_size_sst = args.batch_size // args.gradient_accumulations_sst
    args.gradient_accumulations_para = int(np.ceil(args.batch_size / args.max_batch_size_para))
    args.batch_size_para = args.batch_size // args.gradient_accumulations_para
    args.gradient_accumulations_sts = int(np.ceil(args.batch_size / args.max_batch_size_sts))
    args.batch_size_sts = args.batch_size // args.gradient_accumulations_sts

    # Display some infos here (e.g. batch size, learning rate, etc.)
    print_length = 62
    print_subset_of_args(args, "DATASETS", ["sst_train", "sst_dev", "sst_test", "para_train", "para_dev", "para_test", "sts_train", "sts_dev", "sts_test"], color = Colors.BLUE, print_length = print_length, var_length = 20)
    print_subset_of_args(args, "OUTPUTS", ["sst_dev_out", "sst_test_out", "para_dev_out", "para_test_out", "sts_dev_out", "sts_test_out"], color = Colors.RED, print_length = print_length, var_length = 20)
    print_subset_of_args(args, "PRETRAIING", ["option", "pretrained_model_name"], color = Colors.CYAN, print_length = print_length, var_length = 25)

    hyperparameters = ["n_hidden_layers", "batch_size", "epochs", "lr", "hidden_dropout_prob", "seed"]
    if args.option == "finetune": hyperparameters += ["num_batches_per_epoch"]
    print_subset_of_args(args, "HYPERPARAMETERS", hyperparameters, color = Colors.GREEN, print_length = print_length, var_length = 30)
    
    optim_args = ["use_amp", "use_gpu", "gradient_accumulations_sst", "gradient_accumulations_para", "gradient_accumulations_sts", "patience"]
    if args.option == "finetune":
        optim_args += ["task_scheduler", "projection"]
        if args.projection == "vaccine":
            optim_args += ["beta_vaccine"]
    print_subset_of_args(args, "OPTIMIZATIONS", optim_args, color = Colors.YELLOW, print_length = print_length, var_length = 35)
    print("")

    if args.use_amp and not args.use_gpu:
        raise ValueError("Mixed precision training is only supported on GPU")

    # If we are in testing mode, we do not need to train the model
    if args.option == "test":
        if args.pretrained_model_name == "none":
            raise ValueError("Testing mode requires a pretrained model")
        if args.lr != 1e-5:
            warn("Testing mode does not train the model, so the learning rate is not used")
        if args.epochs != 1:
            warn("Testing mode does not train the model, so the number of epochs is not used")
        if args.num_batches_per_epoch != -1:
            warn("Testing mode does not train the model, so num_batches_per_epoch is not used")
        if args.task_scheduler != "round_robin":
            warn("Testing mode does not train the model, so task_scheduler is not used")
        if args.projection != "none":
            warn("Testing mode does not train the model, so projection is not used")
        if args.hidden_dropout_prob != 0.3:
            warn("Testing mode does not train the model, so hidden_dropout_prob is not used")
        if args.beta_vaccine != 1e-2:
            warn("Testing mode does not train the model, so beta_vaccine is not used")
        if args.patience != 5:
            warn("Testing mode does not train the model, so patience is not used")
        if args.use_amp:
            warn("Testing mode does not train the model, so use_amp is not used")

    # If we are individually pretraining, a lot of options are not available
    elif args.option == "individual_pretrain":
        if args.pretrained_model_name != "none":
            warn("Pretraining mode should not be used with an already pretrained model", color=Colors.YELLOW)
        if args.task_scheduler != "round_robin":
            warn("Pretraining mode does not support task scheduler (Each task is trained separately)")
        if args.projection != "none":
            warn("Pretraining mode does not support projection (Each task is trained separately)")
        if args.num_batches_per_epoch != -1:
            warn("Pretraining mode does not support num_batches_per_epoch (One peoch is a full pass through the dataset)")
        if args.beta_vaccine != 1e-2:
            warn("Pretraining mode does not support beta_vaccine (Each task is trained separately)")
        
    # If we are in finetuning mode or multitask pretraining
    else:
        if args.projection != "vaccine" and args.beta_vaccine != 1e-2:
            warn("Beta for Vaccine is only used when Vaccine is used")
        if args.projection != "none" and args.task_scheduler != "round_robin":
            warn("PCGrad & Vaccine do not use task scheduler")

    return args



if __name__ == "__main__":
    args = get_args()
    args.filepath = args.log_dir + "/best_model.pt" # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    if args.option != "test": train_multitask(args)
    if args.option == "test": args.filepath = args.pretrained_model_name
    if args.option != "pretrain" and args.option != 'individual_pretrain': test_model(args)
