python run_mlm_no_trainer.py ^
--model_name_or_path roberta-base ^
--output_dir ../finetuned_model/roberta_base_warmup_5_bookcorpus ^
--per_device_train_batch_size 8 ^
--num_train_epochs 1 ^
--dataset_name bookcorpus ^
--lr_scheduler_type constant_with_warmup