python run_mlm_no_trainer.py ^
--model_name_or_path roberta-base ^
--train_file ../input/commonlitreadabilityprize/mlm_data.csv ^
--validation_file ../input/commonlitreadabilityprize/mlm_data_val.csv ^
--output_dir ../finetuned_model/roberta_base_warmup_5 ^
--per_device_train_batch_size 16 ^
--num_train_epochs 5 ^
--lr_scheduler_type constant_with_warmup