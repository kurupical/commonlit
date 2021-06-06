python run_mlm_no_trainer.py ^
--model_name_or_path albert-base-v2 ^
--train_file ../input/commonlitreadabilityprize/mlm_data.csv ^
--validation_file ../input/commonlitreadabilityprize/mlm_data_val.csv ^
--output_dir ../finetuned_model/albert_base_v2_15 ^
--per_device_train_batch_size 8 ^
--num_train_epochs 15 ^
--lr_scheduler_type constant_with_warmup