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
