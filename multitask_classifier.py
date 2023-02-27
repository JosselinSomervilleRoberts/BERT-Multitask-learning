import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_multitask, test_model_multitask


TQDM_DISABLE = False

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
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        
        # Step 2: Add a linear layer for sentiment classification
        # For the baseline:
        #   - Calls forward() to get the BERT embeddings
        #   - Applies a dropout layer
        #   - Applies a linear layer to get the logits
        self.dropout_sentiment = nn.Dropout(config.hidden_dropout_prob)
        self.linear_sentiment = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)

        # Step 3: Add a linear layer for paraphrase detection
        self.linear_paraphrase = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)

        # Step 4: Add a linear layer for semantic textual similarity
        self.linear_similarity = nn.Linear(2 * BERT_HIDDEN_SIZE, N_STS_CLASSES)


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


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        # Step 1: Get the BERT embeddings
        cls_embeddings = self.forward(input_ids, attention_mask)

        # Step 2: Get the logits for sentiment classification
        cls_embeddings = self.dropout_sentiment(cls_embeddings)
        logits = self.linear_sentiment(cls_embeddings)

        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # Step 1: Get the BERT embeddings
        cls_embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        # Step 2: Get the logits for paraphrase detection
        cls_embeddings = torch.cat((cls_embeddings_1, cls_embeddings_2), dim=1)
        logits = self.linear_paraphrase(cls_embeddings)

        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # Step 1: Get the BERT embeddings
        cls_embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        # Step 2: Get the logits for semantic textual similarity
        cls_embeddings = torch.cat((cls_embeddings_1, cls_embeddings_2), dim=1)
        logits = self.linear_similarity(cls_embeddings)

        return logits




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
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # SST: Sentiment classification
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Para: Paraphrase detection
    gradient_accumulation_steps = 1
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=int(args.batch_size / gradient_accumulation_steps),
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=int(args.batch_size / gradient_accumulation_steps),
                                    collate_fn=para_dev_data.collate_fn)

    # STS: Semantic textual similarity
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'pretrained_model_name': args.pretrained_model_name}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    if args.pretrained_model_name != "none":
        config = load_model(model, args.pretrained_model_name)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss_sst, train_loss_para, train_loss_sts = 0, 0, 0
        num_batches_sst, num_batches_para, num_batches_sts = 0, 0, 0

        # STS: Semantic textual similarity
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                              batch['attention_mask_1'],
                                                              batch['token_ids_2'],
                                                              batch['attention_mask_2'],
                                                              batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            
            loss.backward()
            optimizer.step()

            train_loss_sts += loss.item()
            num_batches_sts += 1
            if TQDM_DISABLE: print(f'batch {num_batches_sts}/{len(sts_train_dataloader)} STS - loss: {loss.item()}')

        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                              batch['attention_mask_1'],
                                                              batch['token_ids_2'],
                                                              batch['attention_mask_2'],
                                                              batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            # optimizer.zero_grad()
            preds = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            #print(b_ids_1.shape, b_mask_1.shape, b_ids_2.shape, b_mask_2.shape, b_labels.shape)
            loss = F.binary_cross_entropy_with_logits(preds.view(-1), b_labels.float(), reduction='sum')  * gradient_accumulation_steps / args.batch_size

            loss.backward()

            if (num_batches_para + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss_para += loss.item()
            num_batches_para += 1
            if TQDM_DISABLE: print(f'batch {num_batches_para}/{len(para_train_dataloader)} Para - loss: {loss.item()}')
            #print("BEFORE: Memory allocated:", torch.cuda.memory_allocated(device="cuda:0") / 1024 ** 3, "GB")
            #print(torch.cuda.memory_summary())
            torch.cuda.empty_cache()
            #print("\n\nAFTER: Memory allocated:", torch.cuda.memory_allocated(device="cuda:0") / 1024 ** 3, "GB")
            #print(torch.cuda.memory_summary())
            #print("\n\n\n")


        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss_sst += loss.item()
            num_batches_sst += 1
            if TQDM_DISABLE: print(f'batch {num_batches_sst}/{len(sst_train_dataloader)} SST - loss: {loss.item()}')

        train_loss_sst = train_loss_sst / (num_batches_sst)
        train_loss_para = train_loss_para / (num_batches_para)
        train_loss_sts = train_loss_sts / (num_batches_sts)

        (paraphrase_accuracy, para_y_pred, para_sent_ids,
        sentiment_accuracy,sst_y_pred, sst_sent_ids,
        sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        mean_dev_acc = (paraphrase_accuracy + sentiment_accuracy + sts_corr) / 3

        if mean_dev_acc > best_dev_acc:
            best_dev_acc = mean_dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss sst: {train_loss_sst:.3f}, train loss para: {train_loss_para:.3f}, train loss sts: {train_loss_sts:.3f}")
        print(f"Epoch {epoch}: dev acc sst: {sentiment_accuracy:.3f}, dev acc para: {paraphrase_accuracy:.3f}, dev acc sts: {sts_corr:.3f}")
        print("")



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

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--pretrained_model_name", type=str, default="none")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")
    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
