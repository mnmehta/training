CMD="python -u main_ds.py \
--model_name_or_path='/ilab-data/granite-model/granite.7b.llamaT/llamaT-lab-hf.240328a'
--data_path="/dev/shm/data.jsonl"
--output_dir="/new_data/experiments/ap-fsdp-p00-old-m-ds-2t" \
--num_epochs=1 \
--effective_batch_size=48 \
--learning_rate=2e-5 \
--num_warmup_steps=800 \
--sharding_strategy='HYBRID_SHARD' \
--save_samples=230000 \
--log_level="INFO" \
--seed=19437"
deepspeed --num_nodes 1 \
          --num_gpus 8 \
          --no_local_rank \
          --no_python \
          $CMD 2>&1 | tee log.txt
