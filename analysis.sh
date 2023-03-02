# O HIDDEN LAYER
# PRETRAIN
python3 multitask_classifier.py --use_gpu --use_amp --epochs 3 --lr 1e-3 --batch_size 256 --option individual_pretrain --hidden_dropout_prob 0.2  --projection none --patience 3 --n_hidden_layers 0 --max_batch_size_sst 256 --max_batch_size_para 256 --max_batch_size_sts 256 > pretrain_0_logs.txt
mv individual_pretrain-3-0.001-multitask.pt pretrain_0.pt

# FINTETUNE - RANDOM
python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_0.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler random --hidden_dropout_prob 0.2  --projection none --patience 3 --n_hidden_layers 0 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_0_random_logs.txt --save_loss_logs True
mv finetune-10-1e-05-multitask.pt finetune_0_random.pt

# FINTETUNE - ROUND ROBIN
python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_0.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler round_robin --hidden_dropout_prob 0.2  --projection none --patience 3 --n_hidden_layers 0 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_0_round_robin_logs.txt --save_loss_logs True
mv finetune-10-1e-05-multitask.pt finetune_0_round_robin.pt

# FINTETUNE - PAL
python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_0.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler pal --hidden_dropout_prob 0.2  --projection none --patience 3 --n_hidden_layers 0 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_0_pal_logs.txt --save_loss_logs True
mv finetune-10-1e-05-multitask.pt finetune_0_pal.pt

# FINTETUNE - SURGERY
# python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_0.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler pal --hidden_dropout_prob 0.2  --projection pcgrad --patience 3 --n_hidden_layers 0 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_0_surgery_logs.txt 
# mv finetune-10-1e-05-multitask.pt finetune_0_surgery.pt

# FINTETUNE - VACCINE
# python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_0.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler pal --hidden_dropout_prob 0.2  --projection vacce --patience 3 --n_hidden_layers 0 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_0_vaccine_logs.txt 
# mv finetune-10-1e-05-multitask.pt finetune_0_vaccine.pt



# 1 HIDDEN LAYER
# PRETRAIN
python3 multitask_classifier.py --use_gpu --use_amp --epochs 3 --lr 1e-3 --batch_size 256 --option individual_pretrain --hidden_dropout_prob 0.2  --projection none --patience 3 --n_hidden_layers 1 --max_batch_size_sst 256 --max_batch_size_para 256 --max_batch_size_sts 256 > pretrain_1_logs.txt
mv individual_pretrain-3-0.001-multitask.pt pretrain_1.pt

# FINTETUNE - RANDOM
python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_1.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler random --hidden_dropout_prob 0.2  --projection none --patience 3 --n_hidden_layers 1 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_1_random_logs.txt --save_loss_logs True
mv finetune-10-1e-05-multitask.pt finetune_1_random.pt

# FINTETUNE - ROUND ROBIN
python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_1.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler round_robin --hidden_dropout_prob 0.2  --projection none --patience 3 --n_hidden_layers 1 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_1_round_robin_logs.txt --save_loss_logs True
mv finetune-10-1e-05-multitask.pt finetune_1_round_robin.pt

# FINTETUNE - PAL
python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_1.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler pal --hidden_dropout_prob 0.2  --projection none --patience 3 --n_hidden_layers 1 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_1_pal_logs.txt --save_loss_logs True
mv finetune-10-1e-05-multitask.pt finetune_1_pal.pt

# FINTETUNE - SURGERY
# python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_1.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler pal --hidden_dropout_prob 0.2  --projection pcgrad --patience 3 --n_hidden_layers 1 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_1_surgery_logs.txt
# mv finetune-10-1e-05-multitask.pt finetune_1_surgery.pt

# FINTETUNE - VACCINE
# python3 multitask_classifier.py --num_batches_per_epoch 900 --pretrained_model_name pretrain_1.pt --use_gpu --use_amp --epochs 10 --lr 1e-5 --batch_size 128 --option finetune --task_scheduler pal --hidden_dropout_prob 0.2  --projection vacce --patience 3 --n_hidden_layers 1 --max_batch_size_sst 64 --max_batch_size_para 32 --max_batch_size_sts 64 > finetune_1_vaccine_logs.txt
# mv finetune-10-1e-05-multitask.pt finetune_1_vaccine.pt
