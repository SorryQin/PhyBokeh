# === B1: 3t 无 PISA ===
ACCELERATE_USE_CPU=0 \
CUDA_VISIBLE_DEVICES=1 \
WORLD_SIZE=1 RANK=0 MASTER_PORT=29501 \
nohup python train-3t.py --train_data_dir ./temp_data_on_the_fly/ \
  --pretrained_model_name_or_path ./models/sdxl-base \
  --train_batch_size 2 \
  --output_dir ./output/run_ablation_no_pisa_3t \
  --max_train_steps 70000 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision no \
  --learning_rate 3e-5 \
  --lr_scheduler cosine_with_restarts --lr_num_cycles 2 --lr_warmup_steps 20 \
  --resolution 512 \
  --checkpointing_steps 10000 \
  --gan_loss_type multilevel_sigmoid --lambda_gan 1.0 --gan_step 40000 \
  --lpips --edge --lambda_lpips 2 \
  --rank 8 \
  --ablation1 \
  2>&1 | tee ./output/run_ablation_no_pisa_3t/train.log &

# === B2: 5t 无 PISA ===
ACCELERATE_USE_CPU=0 \
CUDA_VISIBLE_DEVICES=2 \
WORLD_SIZE=1 RANK=0 MASTER_PORT=29502 \
nohup python train-5t.py --train_data_dir ./temp_data_on_the_fly/ \
  --pretrained_model_name_or_path ./models/sdxl-base \
  --train_batch_size 2 \
  --output_dir ./output/run_ablation_no_pisa_5t \
  --max_train_steps 70000 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision no \
  --learning_rate 3e-5 \
  --lr_scheduler cosine_with_restarts --lr_num_cycles 2 --lr_warmup_steps 20 \
  --resolution 512 --checkpointing_steps 10000 \
  --gan_loss_type multilevel_sigmoid --lambda_gan 1.0 --gan_step 40000 \
  --lpips --edge --lambda_lpips 2 \
  --rank 8 \
  --ablation1 \
  2>&1 | tee ./output/run_ablation_no_pisa_5t/train.log &

  # === C1: 3t / PISA 无衰减（恒定强度 1.0） ===
ACCELERATE_USE_CPU=0 \
CUDA_VISIBLE_DEVICES=3 \
WORLD_SIZE=1 RANK=0 MASTER_PORT=29503 \
nohup python train-3t.py --train_data_dir ./temp_data_on_the_fly/ \
  --pretrained_model_name_or_path ./models/sdxl-base \
  --train_batch_size 2 \
  --output_dir ./output/run_ablation_no_decay_3t \
  --max_train_steps 70000 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision no \
  --learning_rate 3e-5 \
  --lr_scheduler cosine_with_restarts --lr_num_cycles 2 --lr_warmup_steps 20 \
  --resolution 512 --checkpointing_steps 10000 \
  --gan_loss_type multilevel_sigmoid --lambda_gan 1.0 --gan_step 40000 \
  --lpips --edge --lambda_lpips 2 \
  --rank 8 \
  --pisa_ratio_start 1.0 --pisa_ratio_end 1.0 \
  2>&1 | tee ./output/run_ablation_no_decay_3t/train.log &

# === C2: 5t / PISA 无衰减（恒定强度 1.0） ===
ACCELERATE_USE_CPU=0 \
CUDA_VISIBLE_DEVICES=4 \
WORLD_SIZE=1 RANK=0 MASTER_PORT=29504 \
nohup python train-5t.py --train_data_dir ./temp_data_on_the_fly/ \
  --pretrained_model_name_or_path ./models/sdxl-base \
  --train_batch_size 2 \
  --output_dir ./output/run_ablation_no_decay_5t \
  --max_train_steps 70000 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision no \
  --learning_rate 3e-5 \
  --lr_scheduler cosine_with_restarts --lr_num_cycles 2 --lr_warmup_steps 20 \
  --resolution 512 --checkpointing_steps 10000 \
  --gan_loss_type multilevel_sigmoid --lambda_gan 1.0 --gan_step 40000 \
  --lpips --edge --lambda_lpips 2 \
  --rank 8 \
  --pisa_ratio_start 1.0 --pisa_ratio_end 1.0 \
  2>&1 | tee ./output/run_ablation_no_decay_5t/train.log &
