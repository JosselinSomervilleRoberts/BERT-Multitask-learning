# Bert Multitask Classifier

This project aims to implement a multitask NLP model based on HuggingFace's Bert model `bert-base-uncased` (110M parameters) to perform three downstream task: **sentiment analysis** (multitask classification), **paraphrase detection** (binary classification) and **semantic textual similarity prediction** (regression).

Our model was trained on the following datasets:
- **SST-5**: Stanford Sentiment Treebank.
- **SemEval STS Benchmark Dataset**: for Semantic Textual Similarity
- **Quora Dataset**: question pairs (paraphrase).

## Results
This project was the default final project of the class CS224N at Stanford for the year 2022-2023. You can find the handout [here](https://web.stanford.edu/class/cs224n/project/default-final-project-bert-handout.pdf).

***We finished first (both on the Dev and the Test set) amonst approximately 140 teams.***

## Running the code
We grouped everything ine one script `multitask_classifier.py`. You can call this script with many parameters to choose what you want to do. Here is a quick overview of the most important parameters (Feel free to run `python multitask_classifier.py --help` to get more info):
- `--option [OPTION]`, with `[OPTION] = test, pretrain, individual_pretrain, finetune`. Chosses what you want to do with the model.
- `--use_gpu`, `--use_amp` to run the code on GPU and use AMP (Automated Mixed Precision) in order to have the best performances.
- `--pretrained_model_name [PATH]` with `[PATH]` the path to you `*.pt` file containing the model.
- `--save_path [PATH]` to save the logs, model, predictions and the command to recreate the experiment. We recommand locating your file in `runs/[YOUR-EXPERIMENT-NAME]` to use Tensorboard.
- `--task_scheduler [SCHEDULER]` to choose how to schedule the tasks during training. Options include `random`, `round_robin` or `pal` (recommended).
- `--projection [PROJECTION]` to choose how to handle competing gradients from different tasks. Options include `none`, `pcgrad` or `vaccine`. *(Make sure to use `--combine_strategy [STRATEGY]` if you plan to use a scheduler as weel as a projection. We recommend `encourage`).
- `--use_pal` to add Projected Attention Layers (PAL).
- Some hyperparameters: `--lr [LR]` for the learning rate, `--epochs [EPOCHS]` for the number of epochs, `--num_batches_per_epoch [NB]` for the number of batches per epoch *(Not always apllicable depending on your scheduler)*, `--batch_size [BATCH-SIZE]` for the batch_size, `--batch_size_sts [BATCH-SIZE]` (valid for `sts`, `sst` and `para`) to reduce the batch size of a specific task using **gradient accumulations**.
- More arguments using `--help`.

## Running tensorboard

### Data available
Many metrics are logged to tensorboard:
- **Loss [TASK]** plots the loss of a task after training for x inputs *for all tasks*.
- **Specific Loss [TASK]** plots the loss of a task after training for x inputs *for that task*
- **Dev accuracy [TASK]** plots the accuracy on the Dev set after x inputs on the task.
- **Dev accuracy mean** plots the arithmetic mean of the accuracry for all tasks on the Dev set.
- **[EPOCH] Loss [TASK]** plots the loss per epoch *for that task* (not always relevant as the size of one epoch can be defined by `--num_batches_per_epoch`).
- **Confusion Matrix [TASK]** shows the confusion matrix (available for SST and Paraphrase detection).

### Running tensorboard
The logs are located in your `--save_path`, we recommand using `runs/[YOUR-EXPERIMENT-NAME]`. You can then run:
```bash
tensorboard --logdir=./runs/ --port=6006
```
If you are running this on AWS EC, follow these steps:
1. Run `python multitask_classifier.py` to do your training.
2. Open another SSH console and run: `tensorboard --logdir=./runs/ --port=6006 --bind_all` on your EC2 instance.
3. Open a local terminal and create an SSH tunnel: `ssh -i <YOUR-KEY-PATH> -NL 6006:localhost:6006 ec2-user@<YOUR-PUBLIC-IP>`
4. In the browser, visit: `http://localhost:6006/`

## Using our models
Our trained models are located in `models/`. To download them you will need `git-lfs`. You cna install `git-lfs` locally by running *(source [here](https://stackoverflow.com/questions/48734119/git-lfs-is-not-a-git-command-unclear))*
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs 
```

On AWS EC2, follow the steps detailed [here](https://stackoverflow.com/questions/71448559/git-large-file-storage-how-to-install-git-lfs-on-aws-ec2-linux-2-no-package).

Our models available are:
- `Bert_hidden1.pt` a finetuned model with 1 hidden layer for each task. You can try it with: `python multitask_classifier.py --use_gpu --option test -n_hidden_layers 1 --pretrained_model_name models/Bert_hidden1.pt --save_path runs/Bert_hidden1`.
- `BertPal_hidden1.pt` a finetuned model with 1 hidden layer for each task and some Projected Attention Layers (PAL) of size 128 for each task. You can try it with: `python multitask_classifier.py --use_gpu --option test -n_hidden_layers 1 --pretrained_model_name models/BertPal_hidden1.pt --save_path runs/BertPal_hidden1 --use_pal`.
- `Bert_pretrained1.pt` a pretrained model with 1 hidden layer for each task (the BERT layers are untouched and are the ones from HuggingFace's `bert-base-uncased`). You can try it with: `python multitask_classifier.py --use_gpu --option test -n_hidden_layers 1 --pretrained_model_name models/Bert_pretrained1.pt --save_path runs/Bert_pretrained1`.
- `Bert_fast_finetuned1.pt` a finetuned model with 1 hidden layer for each task. I was trained using a **pal scheduler** as well as **gradient vaccine** with our **NEW COMBINE STRATEGY: encourage**. You can try it with: `python multitask_classifier.py --use_gpu --option test -n_hidden_layers 1 --pretrained_model_name models/Bert_fast_finetuned1.pt --save_path runs/Bert_fast_finetuned1`.

The performances of the models are summarized here:
| Model                | SST     | Para    | STS     | Mean    |
| --------             | ------- | ------- | ------- | ------- |
| Bert_hidden1         | 0.52044 | 0.88319 | 0.87146 | 0.75836 |
| BertPal_hidden1      | TODO    | TODO    | TODO    | TODO    |
| Bert_pretrained1     | TODO    | TODO    | TODO    | TODO    |
| Bert_fast_finetuned1 | TODO    | TODO    | TODO    | TODO    | 