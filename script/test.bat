python run_mlm_no_trainer.py ^
--model_name_or_path bert-base-cased ^
--dataset_name wikitext ^
--dataset_config_name wikitext-2-raw-v1 ^
--output_dir ../finetuned_model/roberta_default ^
--per_device_train_batch_size 16 ^
--num_train_epochs 15