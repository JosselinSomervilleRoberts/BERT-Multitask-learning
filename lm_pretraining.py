import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import DataLoader

from preprocessing.datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data

# Arguments
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--gamma', type=float, default=0.95)
args = parser.parse_args()

args.batch_size_sst = args.batch_size
args.batch_size_para = args.batch_size
args.batch_size_sts = args.batch_size

# Initialize tokenizer and BERT model for MLM pretraining
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Define MLM objective function
criterion = nn.CrossEntropyLoss()

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

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
sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size_sts,
                                    collate_fn=sts_train_data.collate_fn)
sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size_sts,
                                collate_fn=sts_dev_data.collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
model.to(device)

# MLM pretraining loop
for epoch in range(args.num_epochs):

    # MLM objective loop
    for batch_idx, batch in enumerate(para_train_dataloader):
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)

        batch_ids = b_ids_1
        input_mask = (batch_ids != tokenizer.pad_token_id).to(device)
        labels = batch_ids.clone()
        
        # Mask out some tokens at random
        masked_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).bool().to(device)
        masked_indices &= input_mask
        labels[~masked_indices] = -100
        masked_tokens = batch_ids[masked_indices]
        batch_ids[masked_indices] = tokenizer.mask_token_id
        
        # Compute MLM loss and update weights
        optimizer.zero_grad()
        outputs = model(batch_ids, attention_mask=input_mask, labels=labels)
        print("outputs.logits.shape: ", outputs.logits.shape)
        print("masked_tokens.shape: ", masked_tokens.shape)
        loss = criterion(outputs.logits.view(-1, tokenizer.vocab_size), masked_tokens)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch_ids), len(para_train_dataloader.dataset),
                100. * batch_idx / len(para_train_dataloader), loss.item()))
    
    # Update the learning rate
    scheduler.step()