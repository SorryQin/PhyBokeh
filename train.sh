ACCELERATE_USE_CPU=0 \
CUDA_VISIBLE_DEVICES=0 \
WORLD_SIZE=1 \
RANK=0 \
MASTER_PORT=29500 \
nohup python train-lora-new-new.py --train_data_dir ./temp_data_on_the_fly/ \
  --pretrained_model_name_or_path ./models/sdxl-base \
  --train_batch_size 2 \
  --output_dir ./output/run6 \
  --max_train_steps 70000 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision no \
  --learning_rate 3e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_num_cycles 2 \
  --lr_power 0.5 \
  --lr_warmup_steps 20 \
  --resolution 512 \
  --checkpointing_steps 10000 \
  --gan_loss_type multilevel_sigmoid \
  --lambda_gan 1.0 \
  --gan_step 40000 \
  --lpips \
  --edge \
  --lambda_lpips 2 \
  --rank 8 \
  2>&1 | tee ./output/run6/train.log &
