# NQ_BERT-DM
The code for ACL2020 paper "Document Modeling with Graph Attention Networks for Multi-grained Machine Reading Comprehension"

**1. Process training/development data**

```sh
python src_nq_memory/create_examples.py 
	--input_pattern "./natural_questions/v1.0/train/nq-train-*.jsonl.gz" \
	--vocab_file ./bert-base-uncased-vocab.txt \
	--do_lower_case \
	--output_dir ./natural_questions/nq_0.03/ \
	--num_threads 24 --include_unknowns 0.03 --max_seq_length 512 --doc_stride 128
```

```bash
python src_nq_memory/create_examples.py \
	--input_pattern "./natural_questions/v1.0/dev/nq-dev-*.jsonl.gz" \
	--vocab_file ./bert-base-uncased-vocab.txt \
	--do_lower_case \
	--output_dir ./natural_questions/nq_0.03/ \
	--num_threads 24 --include_unknowns 0.03 --max_seq_length 512 --doc_stride 128 
```

**2. Training the model with a 4-GPU node**

```bash
python -u -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 9527 ./src_nq/run.py \
	--my_config ./configs/config.json \
	--feature_path ./natural_questions/nq_0.03/ \
	--train_pattern "./natural_questions/v1.0/train/nq-train-??.jsonl.gz" \
	--test_pattern "./natural_questions/v1.0/dev/nq-dev-*.jsonl.gz" \
	--model_dir /path/to/pre-trained_model/ \
	--output_dir /path/to/output_model_dir \
	--do_train --do_predict --train_batch_size 8 --predict_batch_size 8 \
	--learning_rate 2e-5 --num_train_epochs 2
```

