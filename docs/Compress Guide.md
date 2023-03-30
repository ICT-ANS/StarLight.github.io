---
layout: default
title: Compress Guide
nav_order: 3
---
# Guide for compressing your own networks
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Pruning
1. Load your pre-trained network:
```python
  # get YourPretarinedNetwork and load pre-trained weights for it
  model = YourPretarinedNetwork(args).to(device)
  model.load_state_dict(checkpoint['state_dict'])
```

2. Set `config_list` and choose a suitable pruner:
```python
  from lib.algorithms.pytorch.pruning import (TaylorFOWeightFilterPruner, FPGMPruner, AGPPruner)

  # choose a pruner: agp, taylor, or fpgm
  if args.pruner == 'agp':
      config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
      pruner = AGPPruner(
          model,
          config_list,
          optimizer,
          trainer,
          criterion,
          num_iterations=1,
          epochs_per_iteration=1,
          pruning_algorithm='taylorfo',
      )
  elif args.pruner == 'taylor':
      config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
      pruner = TaylorFOWeightFilterPruner(
          model,
          config_list,
          optimizer,
          trainer,
          criterion,
          sparsifying_training_batches=1,
      )
  elif args.pruner == 'fpgm':
      config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
      pruner = FPGMPruner(
          model,
          config_list,
          optimizer,
          dummy_input=torch.rand(1, 3, 64, 64).to(device),
      )
  else:
      raise NotImplementedError
```
* `sparsity` specifies the pruning sparsity, ranging from 0.0 to 1.0. Larger sparsity corresponds to a more lightweight model.
* `op_types` specifies the type of pruned operation and can be either `Conv2d` or `Conv3d`, or both of them.
* `optimizer`, `trainer`, and `criterion` are the same as pre-training your network.

3. Use the pruner to generate the pruning mask
```python
  # generate and export the pruning mask
  pruner.compress()
  pruner.export_model(
    os.path.join(args.save_dir, 'model_masked.pth'), 
    os.path.join(args.save_dir, 'mask.pth')
  )
```
* `model_masked.pth` includes the model weights and the generated pruning mask.
* `mask.pth` only includes the generated pruning mask.

4. Export your pruned model:
```python
  from lib.compression.pytorch import ModelSpeedup

  # initialize a new model instance and load pre-trained weights with the pruning mask
  model = YourPretarinedNetwork(args).to(device)
  model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_masked.pth')))
  masks_file = os.path.join(args.save_dir, 'mask.pth')

  # use the speedup_model() of ModelSpeedup() to automatically export the pruned model
  m_speedup = ModelSpeedup(model, torch.rand(input_shape).to(device), masks_file, device)
  m_speedup.speedup_model()
```
* `input_shape` denotes the shape of your model inputs with `batchsize=1`. 
* This automatic export method is susceptible to errors when unrecognized structures are present in your model. To assist in resolving any bugs that may arise during the pruning process, we have compiled a summary of known issues in our [Bug Summary](https://github.com/ICT-ANS/StarLight).
* If there are too many errors and it's hard to solve, we recommend you to manually export the pruned model by providing the topology structures of networks. Please refer to this [link](https://ict-ans.github.io/StarLight.github.io/docs/Manually%20Export.html) for more details.

5. Fine-tune your pruned model:
* To fine-tune the pruned model, we suggest following your own pre-training process to minimize the performance drop. 
* Since the pruned model has pre-trained weights and fewer parameters, a smaller `learning_rate` may be more effective during the fine-tuning.

## Quantization
1. Load your pre-trained network:
```python
  # get YourPretarinedNetwork and load pre-trained weights for it
  model = YourPretarinedNetwork(args).to(device)
  model.load_state_dict(checkpoint['state_dict'])
```

2. Initialize the dataloader:
```python
  import torchvision.datasets as datasets

  def get_data_loader(args):
      train_dir = os.path.join(args.data, 'train')
      train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
      train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  
      val_dir = os.path.join(args.data, 'val')
      val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
      val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  
      n_train = len(train_dataset)
      indices = list(range(n_train))
      random.shuffle(indices)
      calib_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:args.calib_num])
      calib_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=calib_sampler)
  
      return train_loader, val_loader, calib_loader
  train_loader, val_loader, calib_loader = get_data_loader(args)
```
* `calib_loader` uses a subset from the training dataset to calibrate during subsequent quantization.

3. Specify `quan_mode` and output paths of onnx, trt, and cache:
```python
  onnx_path = os.path.join(args.save_dir, '{}_{}.onnx'.format(args.model, args.quan_mode))
  trt_path = os.path.join(args.save_dir, '{}_{}.trt'.format(args.model, args.quan_mode))
  cache_path = os.path.join(args.save_dir, '{}_{}.cache'.format(args.model, args.quan_mode))

  if args.quan_mode == "int8":
      extra_layer_bit = 8
  elif args.quan_mode == "fp16":
      extra_layer_bit = 16
  elif args.quan_mode == "best":
      extra_layer_bit = -1
  else:
      extra_layer_bit = 32
```

4. Define the `engine` for inference:
```python
  from lib.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

  engine = ModelSpeedupTensorRT(
      model,
      input_shape,
      config=None,
      calib_data_loader=calib_loader,
      batchsize=args.batch_size,
      onnx_path=onnx_path,
      calibration_cache=cache_path,
      extra_layer_bit=extra_layer_bit,
  )
  if not os.path.exists(trt_path):
      engine.compress()
      engine.export_quantized_model(trt_path)
  else:
      engine.load_quantized_model(trt_path)
```

5. Use the `engine` for inference:
```python
  loss, top1, infer_time = validate(engine, val_loader, criterion)
```
* `engine` is similar to the `model` and can be inferred on either GPU or TensorRT. 
* While the `eval()` method is necessary for `model` inference, it is not required for `engine`.
* Inference with `engine` will return both the outputs and the inference time.

## Pruning and Quantization
* After completing the `Pruning` process outlined above, use the pruned model to undergo the `Quantization` process.