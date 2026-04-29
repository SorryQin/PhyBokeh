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

  ## 9
  目前infer 没完全复现 train 的 forward 分布吗？
  train中写了scheduler.set_timesteps(1000)，但是infer中有对应吗？
  disp_coc scale 可能不一致

## 10 3t和5t报错，但是no-k-decay不报错

Traceback (most recent call last):
  File "/data/juicefs_sharing_data/11186867/bokehdiff-master/train-5t.py", line 1270, in <module>
    main()
  File "/data/juicefs_sharing_data/11186867/bokehdiff-master/train-5t.py", line 860, in main
    accelerator.init_trackers("CTRL", config=vars(args))
  File "/opt/conda/envs/phybokeh/lib/python3.10/site-packages/accelerate/accelerator.py", line 908, in _inner
    return PartialState().on_main_process(function)(*args, **kwargs)
  File "/opt/conda/envs/phybokeh/lib/python3.10/site-packages/accelerate/accelerator.py", line 3307, in init_trackers
    tracker.store_init_configuration(config)
  File "/opt/conda/envs/phybokeh/lib/python3.10/site-packages/accelerate/tracking.py", line 89, in execute_on_main_process
    return PartialState().on_main_process(function)(self, *args, **kwargs)
  File "/opt/conda/envs/phybokeh/lib/python3.10/site-packages/accelerate/tracking.py", line 232, in store_init_configuration
    self.writer.add_hparams(values, metric_dict={})
  File "/opt/conda/envs/phybokeh/lib/python3.10/site-packages/torch/utils/tensorboard/writer.py", line 330, in add_hparams
    exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
  File "/opt/conda/envs/phybokeh/lib/python3.10/site-packages/torch/utils/tensorboard/summary.py", line 318, in hparams
    raise ValueError(
ValueError: value should be one of int, float, str, bool, or torch.Tensor
