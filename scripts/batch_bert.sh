python train.py --data_dir  docred_data \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--save_path checkpoints/bert-annotated-3.pt \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 8 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--evaluation_steps 5000 \
--num_labels 4 \
--classifier_lr 1e-4 \
--learning_rate 3e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 50.0 \
--seed 66 \
--num_class 97
