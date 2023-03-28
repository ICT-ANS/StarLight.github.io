---
layout: default
title: Manually Export
nav_order: 3
---


# Manually export your pruned network
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Load your pre-trained network and define the pruner to automatically wrap modules with mask.
```python
    # get YourPretarinedNetwork and load pre-trained weights for it
    model = YourPretarinedNetwork(args).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    # define the pruner to wrap the network 
    config_list = [
    {'sparsity': args.sparsity,
     'op_types': ['Conv2d'],
     'op_names': HSMNet_2DConv_prune_ops}
    ]
    pruner = FPGMPruner(
        model,
        config_list,
        dummy_input=dummy_inputs,
    )
```

## Generate an `inplace_dict`, which specifies the topology of the network.
* Rules for the `inplace_dict`:
  * The data format of the `inplace_dict` are defined as a Python `dict`. Each key is the name of a Conv2D or BN2D layer to be pruned. Each value is a tuple, including the names of all input layers to the key. The input to the first layer is None.
  * Keys of `inplace_dict` only contain the layers to be pruned. Layers without pruning should not be included as keys but can be in the tuple of values if needed. Note that the last layer of a backone usually does not require pruning.
  * For complex network architectures, you can generate the `inplace_dict` using the following code and then manually adjust some structures. 
  * We provide examples of `inplace_dict` in the following.

* Example for generating the `inplace_dict`:
```python
    # take the HSMNet as an example, use the code blow to get the inplace_dict of 2D conv
    last_conv = None
    last_depthwise_conv = None
    last_layer = -1
    downsample_start = None
    for name, _, in model.named_modules():
        output_flag = False
        if 'feature_extraction' in name \
                and 'cbr_unit.2' not in name:
            if len(name.split('.')) > 3 and name[-1] in ['0', '1']:
                output_flag = True
            if 'res_block' in name and len(name.split('.')) < 6 and 'downsample' not in name:
                output_flag = False
            if 'downsample' in name and len(name.split('.')) < 5:
                output_flag = False
            if 'res_block' in name and 'convbnrelu1.cbr_unit.0' in name:
                downsample_start = last_conv
            if 'pyramid_pooling' in name and len(name.split('.')) < 6:
                output_flag = False
            if output_flag:
                if last_conv is None:
                    print('\'%s\': (%s, ),' % (name, last_conv))
                else:
                    if 'downsample' in name and name[-1] == '0':
                        print('\'%s\': (\'%s\', ),' % (name, downsample_start))
                    else:
                        print('\'%s\': (\'%s\', ),' % (name, last_conv))
                if name[-1] == '0':
                    last_conv = name
```

* Results of the generated inplace_dict:
```python
    inplace_dict = {
        # feature extraction
        'feature_extraction.convbnrelu1_1.cbr_unit.0': (None,),
        'feature_extraction.convbnrelu1_1.cbr_unit.1': ('feature_extraction.convbnrelu1_1.cbr_unit.0',),
        'feature_extraction.convbnrelu1_2.cbr_unit.0': ('feature_extraction.convbnrelu1_1.cbr_unit.0',),
        'feature_extraction.convbnrelu1_2.cbr_unit.1': ('feature_extraction.convbnrelu1_2.cbr_unit.0',),
        'feature_extraction.convbnrelu1_3.cbr_unit.0': ('feature_extraction.convbnrelu1_2.cbr_unit.0',),
        'feature_extraction.convbnrelu1_3.cbr_unit.1': ('feature_extraction.convbnrelu1_3.cbr_unit.0',),
        'feature_extraction.res_block3.0.convbnrelu1.cbr_unit.0': ('feature_extraction.convbnrelu1_3.cbr_unit.0',),
        'feature_extraction.res_block3.0.convbnrelu1.cbr_unit.1': (
            'feature_extraction.res_block3.0.convbnrelu1.cbr_unit.0',),
        'feature_extraction.res_block3.0.convbn2.cb_unit.0': ('feature_extraction.res_block3.0.convbnrelu1.cbr_unit.0',),
        'feature_extraction.res_block3.0.convbn2.cb_unit.1': ('feature_extraction.res_block3.0.convbn2.cb_unit.0',),
        'feature_extraction.res_block3.0.downsample.0': ('feature_extraction.convbnrelu1_3.cbr_unit.0',),
        'feature_extraction.res_block3.0.downsample.1': ('feature_extraction.res_block3.0.downsample.0',),
        'feature_extraction.res_block5.0.convbnrelu1.cbr_unit.0': ('feature_extraction.res_block3.0.downsample.0',),
        'feature_extraction.res_block5.0.convbnrelu1.cbr_unit.1': (
            'feature_extraction.res_block5.0.convbnrelu1.cbr_unit.0',),
        'feature_extraction.res_block5.0.convbn2.cb_unit.0': ('feature_extraction.res_block5.0.convbnrelu1.cbr_unit.0',),
        'feature_extraction.res_block5.0.convbn2.cb_unit.1': ('feature_extraction.res_block5.0.convbn2.cb_unit.0',),
        'feature_extraction.res_block5.0.downsample.0': ('feature_extraction.res_block3.0.downsample.0',),
        'feature_extraction.res_block5.0.downsample.1': ('feature_extraction.res_block5.0.downsample.0',),
        'feature_extraction.res_block6.0.convbnrelu1.cbr_unit.0': ('feature_extraction.res_block5.0.downsample.0',),
        'feature_extraction.res_block6.0.convbnrelu1.cbr_unit.1': (
            'feature_extraction.res_block6.0.convbnrelu1.cbr_unit.0',),
        'feature_extraction.res_block6.0.convbn2.cb_unit.0': ('feature_extraction.res_block6.0.convbnrelu1.cbr_unit.0',),
        'feature_extraction.res_block6.0.convbn2.cb_unit.1': ('feature_extraction.res_block6.0.convbn2.cb_unit.0',),
        'feature_extraction.res_block6.0.downsample.0': ('feature_extraction.res_block5.0.downsample.0',),
        'feature_extraction.res_block6.0.downsample.1': ('feature_extraction.res_block6.0.downsample.0',),
        'feature_extraction.res_block7.0.convbnrelu1.cbr_unit.0': ('feature_extraction.res_block6.0.downsample.0',),
        'feature_extraction.res_block7.0.convbnrelu1.cbr_unit.1': (
            'feature_extraction.res_block7.0.convbnrelu1.cbr_unit.0',),
        'feature_extraction.res_block7.0.convbn2.cb_unit.0': ('feature_extraction.res_block7.0.convbnrelu1.cbr_unit.0',),
        'feature_extraction.res_block7.0.convbn2.cb_unit.1': ('feature_extraction.res_block7.0.convbn2.cb_unit.0',),
        'feature_extraction.res_block7.0.downsample.0': ('feature_extraction.res_block6.0.downsample.0',),
        'feature_extraction.res_block7.0.downsample.1': ('feature_extraction.res_block7.0.downsample.0',),
        # pyramid_pooling
        'feature_extraction.pyramid_pooling.path_module_list.0.cbr_unit.0': (
            'feature_extraction.res_block7.0.downsample.0',),
        'feature_extraction.pyramid_pooling.path_module_list.0.cbr_unit.1': (
            'feature_extraction.pyramid_pooling.path_module_list.0.cbr_unit.0',),
        'feature_extraction.pyramid_pooling.path_module_list.1.cbr_unit.0': (
            'feature_extraction.res_block7.0.downsample.0',),
        'feature_extraction.pyramid_pooling.path_module_list.1.cbr_unit.1': (
            'feature_extraction.pyramid_pooling.path_module_list.1.cbr_unit.0',),
        'feature_extraction.pyramid_pooling.path_module_list.2.cbr_unit.0': (
            'feature_extraction.res_block7.0.downsample.0',),
        'feature_extraction.pyramid_pooling.path_module_list.2.cbr_unit.1': (
            'feature_extraction.pyramid_pooling.path_module_list.2.cbr_unit.0',),
        'feature_extraction.pyramid_pooling.path_module_list.3.cbr_unit.0': (
            'feature_extraction.res_block7.0.downsample.0',),
        'feature_extraction.pyramid_pooling.path_module_list.3.cbr_unit.1': (
            'feature_extraction.pyramid_pooling.path_module_list.3.cbr_unit.0',),
        # upconv
        'feature_extraction.upconv6.1.cbr_unit.0': ('feature_extraction.pyramid_pooling.path_module_list.3.cbr_unit.0',),
        'feature_extraction.upconv6.1.cbr_unit.1': ('feature_extraction.upconv6.1.cbr_unit.0',),
        'feature_extraction.iconv5.cbr_unit.0': (
            'feature_extraction.upconv6.1.cbr_unit.0', 'feature_extraction.res_block6.0.downsample.0'),
        'feature_extraction.iconv5.cbr_unit.1': ('feature_extraction.iconv5.cbr_unit.0',),
    
        'feature_extraction.upconv5.1.cbr_unit.0': ('feature_extraction.res_block6.0.downsample.0',),
        'feature_extraction.upconv5.1.cbr_unit.1': ('feature_extraction.upconv5.1.cbr_unit.0',),
        'feature_extraction.iconv4.cbr_unit.0': (
            'feature_extraction.upconv5.1.cbr_unit.0', 'feature_extraction.res_block5.0.downsample.0'),
        'feature_extraction.iconv4.cbr_unit.1': ('feature_extraction.iconv4.cbr_unit.0',),
    
        'feature_extraction.upconv4.1.cbr_unit.0': ('feature_extraction.res_block5.0.downsample.0',),
        'feature_extraction.upconv4.1.cbr_unit.1': ('feature_extraction.upconv4.1.cbr_unit.0',),
        'feature_extraction.iconv3.cbr_unit.0': (
            'feature_extraction.upconv4.1.cbr_unit.0', 'feature_extraction.res_block3.0.downsample.0'),
        'feature_extraction.iconv3.cbr_unit.1': ('feature_extraction.iconv3.cbr_unit.0',),
        # project
        'feature_extraction.proj6.cbr_unit.0': ('feature_extraction.pyramid_pooling.path_module_list.3.cbr_unit.0',),
        'feature_extraction.proj6.cbr_unit.1': ('feature_extraction.proj6.cbr_unit.0',),
        'feature_extraction.proj5.cbr_unit.0': ('feature_extraction.iconv5.cbr_unit.0',),
        'feature_extraction.proj5.cbr_unit.1': ('feature_extraction.proj5.cbr_unit.0',),
    
        'feature_extraction.proj4.cbr_unit.0': ('feature_extraction.iconv4.cbr_unit.0',),
        'feature_extraction.proj4.cbr_unit.1': ('feature_extraction.proj4.cbr_unit.0',),
    
        'feature_extraction.proj3.cbr_unit.0': ('feature_extraction.iconv3.cbr_unit.0',),
        'feature_extraction.proj3.cbr_unit.1': ('feature_extraction.proj3.cbr_unit.0',),
        'decoder3.convs.0.downsample.conv1': (None,),
        'decoder3.convs.0.downsample.bn': ('decoder3.convs.0.downsample.conv1',),
    }
```


## Load the generated mask and use the `inplace_dict` to export model.
* Get the pruned model using the mask:
```python
    # load mask and export your network using the following functions  
    mask_pt = torch.load(mask_file, map_location=args.device)
    model = get_pruned_model(pruner, model, mask_pt, sparsity)
```

* Functions for replacing each layer:
```python
    def get_pruned_model(pruner, model, mask_pt, sparsity):
    		"""
        Export a pruned model using the generated mask and inplace_dict.
    
        Parameters
        ----------
        pruner: OneshotPruner
        		FPGMPruner or others
        model : torch.nn.Module
            the pre-trained model wrapped by the pruner
        mask_pt : torch.nn.Module
            the generated masks by the pruner
    		sparsity: float
    				the required sparsity for pruning
        Returns
        -------
        model: torch.nn.Module
            the pruned model
        """
        prune_dict = {}
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'weight_mask'):
                    module.weight_mask = mask_pt[name]['weight']
                    module.bias_mask = mask_pt[name]['bias']
                    prune_dict[module.name] = torch.mean(module.weight_mask, dim=(1, 2, 3)).int().cpu().numpy().tolist()
                elif '_se_expand.conv' in name and 'module' not in name:
                    # Handle the SE module
                    depthwise_countpart = mask_pt[name.replace('_se_expand', '_depthwise_conv')]['weight']
                    prune_dict[name] = torch.mean(depthwise_countpart, dim=(1, 2, 3)).int().cpu().numpy().tolist()
                else:
                    pass
            pruner._unwrap_model()
            model = copy.deepcopy(pruner.bound_model)
            pruner._wrap_model()
            for name, module in model.named_modules():
                if name in inplace_dict:
                    print(name, type(module))
                    device = module.weight.device
                    super_module, leaf_module = get_module_by_name(model, name)
                    if type(module) == nn.BatchNorm2d:
                        mask = prune_dict[inplace_dict[name][0]]
                        mask = torch.Tensor(mask).long().to(device)
                        # if 'proj' in name:
                        #     continue
                        compressed_module = replace_batchnorm2d(leaf_module, mask)
                    if type(module) == nn.Conv2d:
                        if inplace_dict[name][0] is None:
                            input_mask = None
                        else:
                            input_mask = []
                            for x in inplace_dict[name]:
                                if type(x) is int:
                                    input_mask += [1] * x
                                else:
                                    input_mask += prune_dict[x]
                        # output_mask = None if name not in prune_dict or 'proj' in name else prune_dict[name]
                        output_mask = None if name not in prune_dict else prune_dict[name]
    
                        # Process downsample_2d_conv in between 3D conv
                        if name == 'decoder3.convs.0.downsample.conv1':
                            in_channels = module.weight.size(1)
                            input_mask = np.ones(in_channels)
                            # zero_index = random.sample(list(range(in_channels)), k=int(math.ceil(in_channels * sparsity)))
                            zero_index = random.sample(list(range(in_channels)), k=int(in_channels * sparsity))
                            input_mask[zero_index] = 0
    
                        if input_mask is not None:
                            input_mask = torch.Tensor(input_mask).long().to(device)
                        if output_mask is not None:
                            output_mask = torch.Tensor(output_mask).long().to(device)
                        compressed_module = replace_conv2d(module, input_mask, output_mask)
                    setattr(super_module, name.split('.')[-1], compressed_module)
        return model
```

* Functions for indexing a module:
```python
    def get_module_by_name(model, module_name):
          """
          Get a module specified by its module name
        Parameters
        ----------
        model : pytorch model
          the pytorch model from which to get its module
        module_name : str
          the name of the required module
        
        Returns
        -------
        module, module
          the parent module of the required module, the required module
        """
        name_list = module_name.split(".")
        for name in name_list[:-1]:
          model = getattr(model, name)
        leaf_module = getattr(model, name_list[-1])
        return model, leaf_module
```

* Functions for getting the reserved channel index:
```python
    def get_index(mask):
        index = []
        for i in range(len(mask)):
          if mask[i] == 1:
              index.append(i)
        return torch.Tensor(index).long().to(mask.device)
``` 

* Functions for replacing BN2D:
```python
    def replace_batchnorm2d(norm, mask):
        """
        Parameters
        ----------
        norm : torch.nn.BatchNorm2d
          The batchnorm module to be replace
        mask : ModuleMasks
          The masks of this module
        
        Returns
        -------
        torch.nn.BatchNorm2d
          The new batchnorm module
        """
        index = get_index(mask)
        num_features = len(index)
        new_norm = torch.nn.BatchNorm2d(num_features=num_features, eps=norm.eps, momentum=norm.momentum, affine=norm.affine, track_running_stats=norm.track_running_stats)
        # assign weights
        new_norm.weight.data = torch.index_select(norm.weight.data, 0, index)
        new_norm.bias.data = torch.index_select(norm.bias.data, 0, index)
        if norm.track_running_stats:
          new_norm.running_mean.data = torch.index_select(norm.running_mean.data, 0, index)
          new_norm.running_var.data = torch.index_select(norm.running_var.data, 0, index)
        return new_norm
``` 

* Functions for replacing Conv2D:
```python
    def replace_conv2d(conv, input_mask, output_mask):
        """
        Parameters
        ----------
        conv : torch.nn.Conv2d
          The conv2d module to be replaced
        mask : ModuleMasks
          The masks of this module
        
        Returns
        -------
        torch.nn.Conv2d
          The new conv2d module
        """
        if input_mask is None:
          in_channels = conv.in_channels
        else:
          in_channels_index = get_index(input_mask)
          in_channels = len(in_channels_index)
        if output_mask is None:
          out_channels = conv.out_channels
        else:
          out_channels_index = get_index(output_mask)
          out_channels = len(out_channels_index)
        
        if conv.groups != 1:
          new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=conv.kernel_size,
                                     stride=conv.stride,
                                     padding=conv.padding,
                                     dilation=conv.dilation,
                                     groups=out_channels,
                                     bias=conv.bias is not None,
                                     padding_mode=conv.padding_mode)
        else:
          new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=conv.kernel_size,
                                     stride=conv.stride,
                                     padding=conv.padding,
                                     dilation=conv.dilation,
                                     groups=conv.groups,
                                     bias=conv.bias is not None,
                                     padding_mode=conv.padding_mode)
        
        new_conv.to(conv.weight.device)
        tmp_weight_data = tmp_bias_data = None
        
        if output_mask is not None:
          tmp_weight_data = torch.index_select(conv.weight.data, 0, out_channels_index)
          if conv.bias is not None:
              tmp_bias_data = torch.index_select(conv.bias.data, 0, out_channels_index)
        else:
          tmp_weight_data = conv.weight.data
        # For the convolutional layers that have more than one group
        # we need to copy the weight group by group, because the input
        # channal is also divided into serveral groups and each group
        # filter may have different input channel indexes.
        input_step = int(conv.in_channels / conv.groups)
        in_channels_group = int(in_channels / conv.groups)
        filter_step = int(out_channels / conv.groups)
        if input_mask is not None:
          if new_conv.groups == out_channels:
              new_conv.weight.data.copy_(tmp_weight_data)
          else:
              for groupid in range(conv.groups):
                  start = groupid * input_step
                  end = (groupid + 1) * input_step
                  current_input_index = list(filter(lambda x: start <= x and x < end, in_channels_index.tolist()))
                  # shift the global index into the group index
                  current_input_index = [x - start for x in current_input_index]
                  # if the groups is larger than 1, the input channels of each
                  # group should be pruned evenly.
                  assert len(current_input_index) == in_channels_group, \
                      'Input channels of each group are not pruned evenly'
                  current_input_index = torch.tensor(current_input_index).to(tmp_weight_data.device)  # pylint: disable=not-callable
                  f_start = groupid * filter_step
                  f_end = (groupid + 1) * filter_step
                  new_conv.weight.data[f_start:f_end] = torch.index_select(tmp_weight_data[f_start:f_end], 1, current_input_index)
        else:
          new_conv.weight.data.copy_(tmp_weight_data)
        
        if conv.bias is not None:
          new_conv.bias.data.copy_(conv.bias.data if tmp_bias_data is None else tmp_bias_data)
        
        return new_conv
  ```