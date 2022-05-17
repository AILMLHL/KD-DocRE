python train.py --data_dir /docred_data/ \
--transformer_type roberta \
--model_name_or_path roberta-large \
--load_path checkpoints/roberta-continue-trained-best.pt \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file dev.json \
--train_batch_size 2 \
--test_batch_size 2 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 2e-5 \
--classifier_lr 1e-4 \
--max_grad_norm 1.0 \
--drop_prob 0.2 \
--warmup_ratio 0.06 \
--start_steps 50000 \
--evaluation_steps 750 \
--num_train_epochs 50.0 \
--seed 99 \
--num_class 97
