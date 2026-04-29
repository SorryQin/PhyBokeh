## 1
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessor2_0 and will be ignored.
## 2
能跑，但是依旧报错
Steps:   0%|          | 0/70000 [00:00<?, ?it/s]cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.
cross_attention_kwargs ['pisa_strength'] are not expected by AttnProcessorDistReciprocal and will be ignored.

## 3
3t（例如 t = 499,300,100）
--pisa_ratio_start 1.0
--pisa_ratio_end 0.3
--step_k_start 1.0
--step_k_end 0.6
直觉上大致会是：

前步：PISA 强、改动大
后步：PISA 仍有 30%，但更新幅度收敛到 60%
5t（例如 499,400,300,200,100）
--pisa_ratio_start 1.0
--pisa_ratio_end 0.2
--step_k_start 1.0
--step_k_end 0.4


## 5
#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=0 python inference-3t.py \
  --pretrained_model_name_or_path "./models/sdxl-base" \
  --resume_from_checkpoint "./output/run6/checkpoint-70000" \
  --test_data_dir "./test_data/input/*.jpg" \
  --organization EBB \
  --output_dir "./infer_outputs" \
  --data_id "run6_3t" \
  --train_T_list 499 300 100 \
  --pisa_ratio_start 1.0 \
  --pisa_ratio_end 0.3 \
  --step_k_start 1.0 \
  --step_k_end 0.6 \
  --mixed_precision no

#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=0 python inference-5t.py \
  --pretrained_model_name_or_path "./models/sdxl-base" \
  --resume_from_checkpoint "./output/run6/checkpoint-70000" \
  --test_data_dir "./test_data/input/*.jpg" \
  --organization EBB \
  --output_dir "./infer_outputs" \
  --data_id "run6_5t" \
  --train_T_list 499 400 300 200 100 \
  --pisa_ratio_start 1.0 \
  --pisa_ratio_end 0.2 \
  --step_k_start 1.0 \
  --step_k_end 0.4 \
  --mixed_precision no

  ## 6
  画质糊了很多，不仅是非主体部分，所有画面都变糊了，而且看起来很毛躁

  ## 7
  现在乱套了，从499到100的图像越来越糊，越来越无序，是不是方向弄反了？

  ## 8
  是因为输入和训练不一样吗？现在的输入的分辨率变大了造成的影响吗？首先就是输出的图像颜色没有原图鲜艳了，然后就是应该对焦的部分，不虚化的部分没有明显的变化，整体的分辨率没降，但是画质下降了很多，看起来就很糊，从499到300再到100，感觉看起来越来越糙

MODEL_PATH="./models/sdxl-base"
COMMON_ARGS="--train_batch_size 2 \
--gradient_accumulation_steps 1 \
--enable_xformers_memory_efficient_attention \
--mixed_precision no \
--lr_scheduler cosine_with_restarts \
--resolution 512"

mkdir -p ./output0428/run1 ./output0428/run2 ./output0428/run3

echo "Starting Task on GPU 1..."
CUDA_VISIBLE_DEVICES=1 \
WORLD_SIZE=1 \
RANK=0 \
nohup python train-3t.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --train_data_dir ./temp_data_on_the_fly/ \
  --output_dir ./output0428/run1 \
  --lr_num_cycles 2 \
  --lr_power 0.5 \
  --lr_warmup_steps 20 \
  --gan_loss_type multilevel_sigmoid \
  --lambda_gan 1.0 \
  --gan_step 40000 \
  --edge --lambda_lpips 2 \
  --learning_rate 3e-5 \
  --checkpointing_steps 10000 \
  --max_train_steps 70000 \
  $COMMON_ARGS \
  > ./output0428/run1/card1.log 2>&1 &


echo "Starting Task on GPU 2..."
CUDA_VISIBLE_DEVICES=2 \
WORLD_SIZE=1 \
RANK=0 \
nohup python train-3t.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --train_data_dir ./temp_data_on_the_fly/ \
  --output_dir ./output0428/run2 \
  --lr_num_cycles 2 \
  --lr_power 0.5 \
  --lr_warmup_steps 20 \
  --gan_loss_type multilevel_sigmoid \
  --lambda_gan 1.0 \
  --gan_step 40000 \
  --edge --lambda_lpips 2 \
  --learning_rate 3e-5 \
  --checkpointing_steps 5000 \
  --max_train_steps 40000 \
  $COMMON_ARGS \
  > ./output0428/run2/card2.log 2>&1 &


echo "Starting Task on GPU 3..."
CUDA_VISIBLE_DEVICES=3 \
WORLD_SIZE=1 \
RANK=0 \
nohup python train-5t.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --train_data_dir ./temp_data_on_the_fly/ \
  --output_dir ./output0428/run3 \
  --lr_num_cycles 2 \
  --lr_power 0.5 \
  --lr_warmup_steps 20 \
  --gan_loss_type multilevel_sigmoid \
  --lambda_gan 1.0 \
  --gan_step 40000 \
  --edge --lambda_lpips 2 \
  --learning_rate 3e-5 \
  --checkpointing_steps 10000 \
  --max_train_steps 70000 \
  $COMMON_ARGS \
  > ./output0428/run3/card3.log 2>&1 &

echo "All tasks submitted. Use 'nvidia-smi' to check GPU usage."

CUDA_VISIBLE_DEVICES=0 python inference-427.py \
    --pretrained_model_name_or_path "./models/sdxl-base" \
    --resume_from_checkpoint "./output0428/run1/checkpoint-20000" \
    --test_data_dir "./test_data/input/*.jpg" \
    --organization EBB \
    --output_dir "./infer-output" \
    --data_id "3t-1-20000" \
    --mixed_precision no \

  ## 9
  目前infer 没完全复现 train 的 forward 分布吗？
  train中写了scheduler.set_timesteps(1000)，但是infer中有对应吗？
  disp_coc scale 可能不一致
