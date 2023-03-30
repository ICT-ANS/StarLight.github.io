---
layout: default
title: Visualization in StarLight
nav_order: 5
---
# Visualize your onw networks in StarLight
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

* Below is a summary of main steps for integration. For more detailed information, please refer to the example provided on [Integrating the ResNet-TinyImageNet to StarLight](https://github.com/ICT-ANS/StarLight).

## Put your dataset and models to the right folder. 
* Put your dataset in the folder of `data/compression/dataset/YOUR_DATASET`.
* Put your original model (without pruning or quantization) in the folder of `data/compression/dataset/inputs/YOUR_DATASET-YOUR_MODEL`. Note that this model should be named as `model.pth`, which includes the model structure and the pre-trained weights, and is acquired by [saving the entire model using PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

## Configure your dataset and models.
* In `compression_vis/config/global.yaml`, according to our provided examples, fill in the following blocks: `dataset,model`, `support_combinations`, `origin_performance`, `figures`.
* In `compression_vis/config/hyperparameters_setting.yaml`, according to our provided examples, fill in the `default_setting` block with the required hyperparameters (`ft_lr`, `ft_bs`, `ft_epochs`, `prune_sparsity`, `gpus`).

## Generate necessary information during your pruning or quantization.
* Before pruning or quantization, add the code below:
```python
  # for pruning or quantization
  if args.write_yaml:
      flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
      _, top1, _, infer_time, _ = validate(model, val_loader, criterion)
      storage = os.path.getsize(args.resume)
      with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
          yaml_data = {
              'Accuracy': {'baseline': round(top1, 2), 'method': None},
              'FLOPs': {'baseline': round(flops/1e6, 2), 'method': None},
              'Parameters': {'baseline': round(params/1e6, 2), 'method': None},
              'Infer_times': {'baseline': round(infer_time*1e3, 2), 'method': None},
              'Storage': {'baseline': round(storage/1e6, 2), 'method': None},
          }
          yaml.dump(yaml_data, f)
```

* After pruning (usually requires fine-tuning) or quantization, add the code below:
```python
  # for pruning: 
  if epoch == args.finetune_epochs - 1:
      if args.write_yaml and not args.no_write_yaml_after_prune:
          storage = os.path.getsize(os.path.join(args.save_dir, 'model_speed_up_finetuned.pth'))
          with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
              yaml_data = {
                  'Accuracy': {'baseline': yaml_data['Accuracy']['baseline'], 'method': round(top1, 2)},
                  'FLOPs': {'baseline': yaml_data['FLOPs']['baseline'], 'method': round(flops/1e6, 2)},
                  'Parameters': {'baseline': yaml_data['Parameters']['baseline'], 'method': round(params/1e6, 2)},
                  'Infer_times': {'baseline': yaml_data['Infer_times']['baseline'], 'method': round(infer_time*1e3, 2)},
                  'Storage': {'baseline': yaml_data['Storage']['baseline'], 'method': round(storage/1e6, 2)},
                  'Output_file': os.path.join(args.save_dir, 'model_speed_up_finetuned.pth'),
              }
              yaml.dump(yaml_data, f)
  
  # for quantization:
  if args.write_yaml:
    storage = os.path.getsize(trt_path)
    with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
        yaml_data = {
            'Accuracy': {'baseline': yaml_data['Accuracy']['baseline'], 'method': round(top1, 2)},
            'FLOPs': {'baseline': yaml_data['FLOPs']['baseline'], 'method': round(flops/1e6, 2)},
            'Parameters': {'baseline': yaml_data['Parameters']['baseline'], 'method': round(params/1e6, 2)},
            'Infer_times': {'baseline': yaml_data['Infer_times']['baseline'], 'method': round(infer_time*1e3, 2)},
            'Storage': {'baseline': yaml_data['Storage']['baseline'], 'method': round(storage/1e6, 2)},
            'Output_file': os.path.join(args.save_dir, '{}_{}.trt'.format(args.model, args.quan_mode)),
        }
        yaml.dump(yaml_data, f)
```

## (Optional) Compress your network in StarLight using the online mode.
* Create a script `compress.sh` in the folder of `algorithms/compression/nets/YOUR_MODEL/shell`.
* Define the required hyper-parameters as below:
```shell
  dataset=$1
  model=$2
  prune_method=$3
  quan_method=$4
  ft_lr=$5
  ft_bs=$6
  ft_epochs=$7
  prune_sparisity=$8
  gpus=$9
  input_path=${10}
  output_path=${11}
  dataset_path=${12}
```
* Use the above hyper-parameters to start your pruning or quantization. Please refer to our provided examples in `algorithms/compression/nets/ResNet50/shell/compress.sh` to write your startup command.

## Visualization of network features.
* Add 6 randomly selected pictures to the folder of `data/compression/quiver/YOUR_DATASET`.
* Specify the resolution of your inputs in the `img_size` block of `compression_vis/config/global.yaml`.
* Add your entired model namely `model.pth` to the folder of `data/compression/model_vis/YOUR_DATASET-YOUR_MODEL`.
* Add your entired pruned model to the folder of `data/compression/model_vis/YOUR_DATASET-YOUR_MODEL`, they should be named as `online-PRUNER.pth` or `offline-PRUNER.pth` for online and offline mode, respectively.