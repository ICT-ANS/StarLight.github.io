---
layout: default
title: Manually Export (3D-Conv)
nav_order: 5
---


# Manually export your network by pruning 3D-Conv
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Load your pre-trained network.
```python
    # get YourPretarinedNetwork and load pre-trained weights for it
    model = YourPretarinedNetwork(args).to(device)
    model.load_state_dict(checkpoint['state_dict'])
```

## Generate an `conv3d_inplace_dict`, which specifies the topology of the network.
* Rules for the `conv3d_inplace_dict`:
  * The data format of the `conv3d_inplace_dict` are defined as a Python `dict`. Each key is the name of a Conv3D or BN3D layer to be pruned. Each value is a tuple, including the names of all input layers to the key. The input to the first layer is None.
  * Keys of `conv3d_inplace_dict` only contain the layers to be pruned. Layers without pruning should not be included as keys but can be in the tuple of values if needed. Note that the last layer of a backone usually does not require pruning.
  * For complex network architectures, you can generate the `conv3d_inplace_dict` using the following code and then manually adjust some structures. 
  * We provide examples of `conv3d_inplace_dict` in the following.

* Example for generating the `conv3d_inplace_dict` is as follows:
```python
    # take the HSMNet as an example, use the code blow to get the inplace_dict of 3D conv
    last_conv = None
    last_depthwise_conv = None
    last_layer = -1
    pool_convs_start = None
    for name, _, in model.named_modules():
        output_flag = False
        if 'feature_extraction' not in name and len(name.split('.')) > 3:
            output_flag = True
        # we observe that 3d conv in this model has ['decoder', 'conv', 'pool'] in name
        if 'decoder' in name and 'conv' in name and 'pool' not in name and len(name.split('.')) < 5:
            output_flag = False
        if 'pool_convs.0.0' in name:
            pool_convs_start = last_conv
        if output_flag:
            if last_conv is None:
                print('\'%s\': (%s, ),' % (name, last_conv))
            else:
                if 'pool_convs' in name and name[-1] == '0':
                    print('\'%s\': (\'%s\', ),' % (name, pool_convs_start))
                else:
                    print('\'%s\': (\'%s\', ),' % (name, last_conv))
            if name != '' and name[-1] == '0':
                last_conv = name
```

* The generated `conv3d_inplace_dict` is as below:
```python
    conv3d_inplace_dict = {
    # decoder6
    'decoder6.convs.0.conv1.0': (None,),
    'decoder6.convs.0.conv1.1': ('decoder6.convs.0.conv1.0',),
    'decoder6.convs.0.conv2.0': ('decoder6.convs.0.conv1.0',),
    'decoder6.convs.0.conv2.1': ('decoder6.convs.0.conv2.0',),
    'decoder6.convs.1.conv1.0': ('decoder6.convs.0.conv2.0',),
    'decoder6.convs.1.conv1.1': ('decoder6.convs.1.conv1.0',),
    'decoder6.convs.1.conv2.0': ('decoder6.convs.1.conv1.0',),
    'decoder6.convs.1.conv2.1': ('decoder6.convs.1.conv2.0',),
    'decoder6.convs.2.conv1.0': ('decoder6.convs.1.conv2.0',),
    'decoder6.convs.2.conv1.1': ('decoder6.convs.2.conv1.0',),
    'decoder6.convs.2.conv2.0': ('decoder6.convs.2.conv1.0',),
    'decoder6.convs.2.conv2.1': ('decoder6.convs.2.conv2.0',),
    'decoder6.convs.3.conv1.0': ('decoder6.convs.2.conv2.0',),
    'decoder6.convs.3.conv1.1': ('decoder6.convs.3.conv1.0',),
    'decoder6.convs.3.conv2.0': ('decoder6.convs.3.conv1.0',),
    'decoder6.convs.3.conv2.1': ('decoder6.convs.3.conv2.0',),
    'decoder6.convs.4.conv1.0': ('decoder6.convs.3.conv2.0',),
    'decoder6.convs.4.conv1.1': ('decoder6.convs.4.conv1.0',),
    'decoder6.convs.4.conv2.0': ('decoder6.convs.4.conv1.0',),
    'decoder6.convs.4.conv2.1': ('decoder6.convs.4.conv2.0',),
    'decoder6.convs.5.conv1.0': ('decoder6.convs.4.conv2.0',),
    'decoder6.convs.5.conv1.1': ('decoder6.convs.5.conv1.0',),
    'decoder6.convs.5.conv2.0': ('decoder6.convs.5.conv1.0',),
    'decoder6.convs.5.conv2.1': ('decoder6.convs.5.conv2.0',),
    'decoder6.pool_convs.0.0': ('decoder6.convs.5.conv2.0',),
    'decoder6.pool_convs.0.1': ('decoder6.pool_convs.0.0',),
    'decoder6.pool_convs.1.0': ('decoder6.convs.5.conv2.0',),
    'decoder6.pool_convs.1.1': ('decoder6.pool_convs.1.0',),
    'decoder6.pool_convs.2.0': ('decoder6.convs.5.conv2.0',),
    'decoder6.pool_convs.2.1': ('decoder6.pool_convs.2.0',),
    'decoder6.pool_convs.3.0': ('decoder6.convs.5.conv2.0',),
    'decoder6.pool_convs.3.1': ('decoder6.pool_convs.3.0',),
    'decoder6.up.1.0': ('decoder6.pool_convs.3.0',),
    'decoder6.up.1.1': ('decoder6.up.1.0',),
    'decoder6.classify.0.0': ('decoder6.pool_convs.3.0',),
    'decoder6.classify.0.1': ('decoder6.classify.0.0',),
    'decoder6.classify.2.0': ('decoder6.classify.0.0',),
    # decoder5
    'decoder5.convs.0.conv1.0': ('decoder6.up.1.0',),
    'decoder5.convs.0.conv1.1': ('decoder5.convs.0.conv1.0',),
    'decoder5.convs.0.conv2.0': ('decoder5.convs.0.conv1.0',),
    'decoder5.convs.0.conv2.1': ('decoder5.convs.0.conv2.0',),
    'decoder5.convs.1.conv1.0': ('decoder5.convs.0.conv2.0',),
    'decoder5.convs.1.conv1.1': ('decoder5.convs.1.conv1.0',),
    'decoder5.convs.1.conv2.0': ('decoder5.convs.1.conv1.0',),
    'decoder5.convs.1.conv2.1': ('decoder5.convs.1.conv2.0',),
    'decoder5.convs.2.conv1.0': ('decoder5.convs.1.conv2.0',),
    'decoder5.convs.2.conv1.1': ('decoder5.convs.2.conv1.0',),
    'decoder5.convs.2.conv2.0': ('decoder5.convs.2.conv1.0',),
    'decoder5.convs.2.conv2.1': ('decoder5.convs.2.conv2.0',),
    'decoder5.convs.3.conv1.0': ('decoder5.convs.2.conv2.0',),
    'decoder5.convs.3.conv1.1': ('decoder5.convs.3.conv1.0',),
    'decoder5.convs.3.conv2.0': ('decoder5.convs.3.conv1.0',),
    'decoder5.convs.3.conv2.1': ('decoder5.convs.3.conv2.0',),
    'decoder5.convs.4.conv1.0': ('decoder5.convs.3.conv2.0',),
    'decoder5.convs.4.conv1.1': ('decoder5.convs.4.conv1.0',),
    'decoder5.convs.4.conv2.0': ('decoder5.convs.4.conv1.0',),
    'decoder5.convs.4.conv2.1': ('decoder5.convs.4.conv2.0',),
    'decoder5.convs.5.conv1.0': ('decoder5.convs.4.conv2.0',),
    'decoder5.convs.5.conv1.1': ('decoder5.convs.5.conv1.0',),
    'decoder5.convs.5.conv2.0': ('decoder5.convs.5.conv1.0',),
    'decoder5.convs.5.conv2.1': ('decoder5.convs.5.conv2.0',),
    'decoder5.pool_convs.0.0': ('decoder5.convs.5.conv2.0',),
    'decoder5.pool_convs.0.1': ('decoder5.pool_convs.0.0',),
    'decoder5.pool_convs.1.0': ('decoder5.convs.5.conv2.0',),
    'decoder5.pool_convs.1.1': ('decoder5.pool_convs.1.0',),
    'decoder5.pool_convs.2.0': ('decoder5.convs.5.conv2.0',),
    'decoder5.pool_convs.2.1': ('decoder5.pool_convs.2.0',),
    'decoder5.pool_convs.3.0': ('decoder5.convs.5.conv2.0',),
    'decoder5.pool_convs.3.1': ('decoder5.pool_convs.3.0',),
    'decoder5.up.1.0': ('decoder5.pool_convs.3.0',),
    'decoder5.up.1.1': ('decoder5.up.1.0',),
    'decoder5.classify.0.0': ('decoder5.pool_convs.3.0',),
    'decoder5.classify.0.1': ('decoder5.classify.0.0',),
    'decoder5.classify.2.0': ('decoder5.classify.0.0',),
    # decoder4
    'decoder4.convs.0.conv1.0': ('decoder5.up.1.0',),
    'decoder4.convs.0.conv1.1': ('decoder4.convs.0.conv1.0',),
    'decoder4.convs.0.conv2.0': ('decoder4.convs.0.conv1.0',),
    'decoder4.convs.0.conv2.1': ('decoder4.convs.0.conv2.0',),
    'decoder4.convs.1.conv1.0': ('decoder4.convs.0.conv2.0',),
    'decoder4.convs.1.conv1.1': ('decoder4.convs.1.conv1.0',),
    'decoder4.convs.1.conv2.0': ('decoder4.convs.1.conv1.0',),
    'decoder4.convs.1.conv2.1': ('decoder4.convs.1.conv2.0',),
    'decoder4.convs.2.conv1.0': ('decoder4.convs.1.conv2.0',),
    'decoder4.convs.2.conv1.1': ('decoder4.convs.2.conv1.0',),
    'decoder4.convs.2.conv2.0': ('decoder4.convs.2.conv1.0',),
    'decoder4.convs.2.conv2.1': ('decoder4.convs.2.conv2.0',),
    'decoder4.convs.3.conv1.0': ('decoder4.convs.2.conv2.0',),
    'decoder4.convs.3.conv1.1': ('decoder4.convs.3.conv1.0',),
    'decoder4.convs.3.conv2.0': ('decoder4.convs.3.conv1.0',),
    'decoder4.convs.3.conv2.1': ('decoder4.convs.3.conv2.0',),
    'decoder4.convs.4.conv1.0': ('decoder4.convs.3.conv2.0',),
    'decoder4.convs.4.conv1.1': ('decoder4.convs.4.conv1.0',),
    'decoder4.convs.4.conv2.0': ('decoder4.convs.4.conv1.0',),
    'decoder4.convs.4.conv2.1': ('decoder4.convs.4.conv2.0',),
    'decoder4.convs.5.conv1.0': ('decoder4.convs.4.conv2.0',),
    'decoder4.convs.5.conv1.1': ('decoder4.convs.5.conv1.0',),
    'decoder4.convs.5.conv2.0': ('decoder4.convs.5.conv1.0',),
    'decoder4.convs.5.conv2.1': ('decoder4.convs.5.conv2.0',),
    'decoder4.up.1.0': ('decoder4.convs.5.conv2.0',),
    'decoder4.up.1.1': ('decoder4.up.1.0',),
    'decoder4.classify.0.0': ('decoder4.convs.5.conv2.0',),
    'decoder4.classify.0.1': ('decoder4.classify.0.0',),
    'decoder4.classify.2.0': ('decoder4.classify.0.0',),
    # decoder3
    'decoder3.convs.0.conv1.0': ('decoder4.up.1.0',),
    'decoder3.convs.0.conv1.1': ('decoder3.convs.0.conv1.0',),
    'decoder3.convs.0.conv2.0': ('decoder3.convs.0.conv1.0',),
    'decoder3.convs.0.conv2.1': ('decoder3.convs.0.conv2.0',),
    'decoder3.convs.1.conv1.0': ('decoder3.convs.0.conv2.0',),
    'decoder3.convs.1.conv1.1': ('decoder3.convs.1.conv1.0',),
    'decoder3.convs.1.conv2.0': ('decoder3.convs.1.conv1.0',),
    'decoder3.convs.1.conv2.1': ('decoder3.convs.1.conv2.0',),
    'decoder3.convs.2.conv1.0': ('decoder3.convs.1.conv2.0',),
    'decoder3.convs.2.conv1.1': ('decoder3.convs.2.conv1.0',),
    'decoder3.convs.2.conv2.0': ('decoder3.convs.2.conv1.0',),
    'decoder3.convs.2.conv2.1': ('decoder3.convs.2.conv2.0',),
    'decoder3.convs.3.conv1.0': ('decoder3.convs.2.conv2.0',),
    'decoder3.convs.3.conv1.1': ('decoder3.convs.3.conv1.0',),
    'decoder3.convs.3.conv2.0': ('decoder3.convs.3.conv1.0',),
    'decoder3.convs.3.conv2.1': ('decoder3.convs.3.conv2.0',),
    'decoder3.convs.4.conv1.0': ('decoder3.convs.3.conv2.0',),
    'decoder3.convs.4.conv1.1': ('decoder3.convs.4.conv1.0',),
    'decoder3.convs.4.conv2.0': ('decoder3.convs.4.conv1.0',),
    'decoder3.convs.4.conv2.1': ('decoder3.convs.4.conv2.0',),
    'decoder3.classify.0.0': ('decoder3.convs.4.conv2.0',),
    'decoder3.classify.0.1': ('decoder3.classify.0.0',),
    'decoder3.classify.2.0': ('decoder3.classify.0.0',),
}
```

## Specify the `prune_ops`:
```python
HSMNet_3DConv_prune_ops = [
    # decoder6
    'decoder6.convs.0.conv1.0', 'decoder6.convs.0.conv2.0', 'decoder6.convs.1.conv1.0',
    'decoder6.convs.1.conv2.0', 'decoder6.convs.2.conv1.0', 'decoder6.convs.2.conv2.0',
    'decoder6.convs.3.conv1.0', 'decoder6.convs.3.conv2.0', 'decoder6.convs.4.conv1.0',
    'decoder6.convs.4.conv2.0', 'decoder6.convs.5.conv1.0', 'decoder6.convs.5.conv2.0',
    'decoder6.pool_convs.0.0', 'decoder6.pool_convs.1.0', 'decoder6.pool_convs.2.0',
    'decoder6.pool_convs.3.0', 'decoder6.up.1.0',
    # decoder5
    'decoder5.convs.0.conv1.0', 'decoder5.convs.0.conv2.0',
    'decoder5.convs.1.conv1.0', 'decoder5.convs.1.conv2.0', 'decoder5.convs.2.conv1.0',
    'decoder5.convs.2.conv2.0', 'decoder5.convs.3.conv1.0', 'decoder5.convs.3.conv2.0',
    'decoder5.convs.4.conv1.0', 'decoder5.convs.4.conv2.0', 'decoder5.convs.5.conv1.0',
    'decoder5.convs.5.conv2.0', 'decoder5.pool_convs.0.0', 'decoder5.pool_convs.1.0',
    'decoder5.pool_convs.2.0', 'decoder5.pool_convs.3.0', 'decoder5.up.1.0',
    # decoder4
    'decoder4.convs.0.conv1.0',
    'decoder4.convs.0.conv2.0', 'decoder4.convs.1.conv1.0', 'decoder4.convs.1.conv2.0',
    'decoder4.convs.2.conv1.0', 'decoder4.convs.2.conv2.0', 'decoder4.convs.3.conv1.0',
    'decoder4.convs.3.conv2.0', 'decoder4.convs.4.conv1.0', 'decoder4.convs.4.conv2.0',
    'decoder4.convs.5.conv1.0', 'decoder4.convs.5.conv2.0', 'decoder4.up.1.0',
    # decoder3
    'decoder3.convs.0.conv1.0',
    'decoder3.convs.0.conv2.0', 'decoder3.convs.1.conv1.0', 'decoder3.convs.1.conv2.0',
    'decoder3.convs.2.conv1.0', 'decoder3.convs.2.conv2.0', 'decoder3.convs.3.conv1.0',
    'decoder3.convs.3.conv2.0', 'decoder3.convs.4.conv1.0', 'decoder3.convs.4.conv2.0'
]
```


## Use the `conv3d_inplace_dict` and `prune_ops` to export model.
* Generate the 3d mask and get the pruned 3d model:
```python
    model = get_pruned3d_model(model, HSMNet_3DConv_prune_ops, sparsity)

    def get_pruned3d_model(model, prune_ops, sparsity):
        # summarize the prune_list
        prune_list = []
        for name, module, in model.named_modules():
            if isinstance(module, nn.Conv3d):
                # print(name)
                prune_list.append(name)
            if isinstance(module, nn.BatchNorm3d):
                # print(name)
                prune_list.append(name)
        print(prune_list)
        ignore_input_list = []
        ignore_output_list = []
        print('Prune_list: %d, ignore_input_list: %d, ingore_output_list:%d' %
              (len(prune_list), len(ignore_input_list), len(ignore_output_list)))

        # analyze each operation
        count = 0
        output_mask_dict = {}
        for name, module in model.named_modules():
            if name in prune_list:
                print(count, name)
                count += 1
                super_module, leaf_module = get_module_by_name(model, name)
                # prune Conv3d
                if type(module) == nn.Conv3d:
                    # generate masks using the random or fpgm method
                    if 'convs.0.conv1.0' in name:
                        input_mask = get_prune3d_masks(
                            sparsity, 'conv_3d', name, module.weight.data, 'random', 'input',
                            ignore_input_list, ignore_output_list, output_mask_dict
                        )
                    else:
                        input_mask = get_prune3d_masks(
                            sparsity, 'conv_3d', name, module.weight.data, 'fpgm', 'input',
                            ignore_input_list, ignore_output_list, output_mask_dict
                        )
                    output_mask = get_prune3d_masks(
                        sparsity, 'conv_3d', name, module.weight.data, 'fpgm', 'output',
                        ignore_input_list, ignore_output_list, output_mask_dict
                    )
                    if name not in prune_ops:
                        output_mask = None
                    if output_mask is not None:
                        output_mask_dict[name] = output_mask.tolist()
                    else:
                        output_mask_dict[name] = output_mask
                    device = module.weight.device
    
                    # Prune 3d-conv with output masks
                    if input_mask is not None:
                        input_mask = torch.Tensor(input_mask).long().to(device)
                    if output_mask is not None:
                        output_mask = torch.Tensor(output_mask).long().to(device)
                    compressed_module = replace_conv3d(module, input_mask, output_mask)
                    setattr(super_module, name.split('.')[-1], compressed_module)
                    # print('Prune 3d_conv: %s done' % name)
                # prune BN3d
                if type(module) == nn.BatchNorm3d:
                    if name in ignore_output_list:
                        continue
                    output_mask = get_prune3d_masks(
                        sparsity, 'conv_3d', name, module.weight.data, 'fpgm', 'input',
                        ignore_input_list, ignore_output_list, output_mask_dict
                    )
                    if output_mask is None:
                        continue
                    device = module.weight.device
                    # Prune with output masks
                    mask = torch.Tensor(output_mask).long().to(device)
                    compressed_module = replace_batchnorm3d(leaf_module, mask)
                    setattr(super_module, name.split('.')[-1], compressed_module)
                    # print('Prune 3d_bn: %s done' % name)
        print('Prune model 3d_conv and 3d_bn done.')
        return model
```

* Functions for generating 3d mask using the random or fpgm method:
```python
def get_prune3d_masks(sparsity, op, name, weight, method, type, ignore_input_list, ignore_output_list, output_mask_dict=None):
    if op == 'conv_3d':
        if type == 'input' and name in ignore_input_list:
            input_mask = None
            return input_mask
        if type == 'output' and name in ignore_output_list:
            output_mask = None
            return output_mask
        if method == 'random':
            if type == 'input':
                in_channels = weight.size(1)
                input_mask = np.ones(in_channels)
                zero_index = random.sample(list(range(in_channels)), k=int(in_channels * sparsity))
                # zero_index = random.sample(list(range(in_channels)), k=int(math.ceil(in_channels * sparsity)))
                input_mask[zero_index] = 0
                return input_mask
            elif type == 'output':
                out_channels = weight.size(0)
                output_mask = np.ones(out_channels)
                # zero_index = random.sample(list(range(out_channels)), k=int(math.ceil(out_channels * sparsity)))
                zero_index = random.sample(list(range(out_channels)), k=int(out_channels * sparsity))
                output_mask[zero_index] = 0
                return output_mask
            else:
                raise NotImplementedError
        elif method == 'fpgm':
            if type == 'input':
                input_mask = []
                for _previous_name in conv3d_topology_dict[name]:
                    if output_mask_dict[_previous_name] is not None:
                        input_mask += output_mask_dict[_previous_name]
                    else:
                        return None
                return input_mask
            elif type == 'output':
                output_mask = get_fpgm_mask(weight, sparsity)
                return output_mask
            else:
                raise NotImplementedError
    elif op == 'bn_3d':
        if method == 'random':
            # Generate random output masks
            out_channels = weight.size(0)
            output_mask = np.ones(out_channels)
            zero_index = random.sample(list(range(out_channels)), k=int(out_channels * sparsity))
            output_mask[zero_index] = 0
        elif method == 'fpgm':
            output_mask = []
            for _previous_name in conv3d_topology_dict[name]:
                output_mask += output_mask_dict[_previous_name]
            return output_mask
        pass

```

* Functions for generating 3d masks via FPGM method:
```python
def get_fpgm_mask(weight, sparsity):
    def get_distance_sum(weight, out_idx):
        """
        Calculate the total distance between a specified filter (by out_idex and in_idx) and
        all other filters.
        Parameters
        ----------
        weight: Tensor
            convolutional filter weight
        out_idx: int
            output channel index of specified filter, this method calculates the total distance
            between this specified filter and all other filters.
        Returns
        -------
        float32
            The total distance
        """
        # print('weight size: %s', weight.size())
        assert len(weight.size()) in [3, 4, 5], 'unsupported weight shape'

        w = weight.view(weight.size(0), -1)
        anchor_w = w[out_idx].unsqueeze(0).expand(w.size(0), w.size(1))
        x = w - anchor_w
        x = (x * x).sum(-1)
        x = torch.sqrt(x)
        return x.sum()

    def get_channel_sum(weight):
        assert len(weight.size()) in [3, 4, 5]
        dist_list = []
        for out_i in range(weight.size(0)):
            dist_sum = get_distance_sum(weight, out_i)
            dist_list.append(dist_sum)
        return torch.Tensor(dist_list).to(weight.device)

    def get_min_gm_kernel_idx(weight, num_prune):
        channel_dist = get_channel_sum(weight)
        dist_list = [(channel_dist[i], i)
                     for i in range(channel_dist.size(0))]
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:num_prune]
        return [x[1] for x in min_gm_kernels]

    out_channels = weight.size(0)
    num_prune = int(out_channels * sparsity)
    # num_prune = int(math.ceil(out_channels * sparsity))
    min_gm_idx = get_min_gm_kernel_idx(weight, num_prune)
    output_mask = np.ones(out_channels)
    output_mask[min_gm_idx] = 0

    return output_mask
``` 

* Functions for replacing Conv3d:
```python
def replace_conv3d(conv, input_mask, output_mask):
    """
    Parameters
    ----------
    conv : torch.nn.Conv3d
        The conv3d module to be replaced
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.Conv3d
        The new conv3d module
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
        new_conv = torch.nn.Conv3d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   dilation=conv.dilation,
                                   groups=out_channels,
                                   bias=conv.bias is not None,
                                   padding_mode=conv.padding_mode)
    else:
        new_conv = torch.nn.Conv3d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   dilation=conv.dilation,
                                   groups=conv.groups,
                                   bias=conv.bias is not None,
                                   padding_mode=conv.padding_mode)
    # print(new_conv.weight.data.shape)
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
        if new_conv.groups == out_channels and out_channels != 1:
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

* Functions for replacing BN3d:
```python
def replace_batchnorm3d(norm, mask):
    """
    Parameters
    ----------
    norm : torch.nn.BatchNorm3d
        The batchnorm module to be replace
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.BatchNorm3d
        The new batchnorm module
    """
    index = get_index(mask)
    num_features = len(index)
    new_norm = torch.nn.BatchNorm3d(num_features=num_features, eps=norm.eps, momentum=norm.momentum, affine=norm.affine, track_running_stats=norm.track_running_stats)
    # assign weights
    new_norm.weight.data = torch.index_select(norm.weight.data, 0, index)
    new_norm.bias.data = torch.index_select(norm.bias.data, 0, index)
    if norm.track_running_stats:
        new_norm.running_mean.data = torch.index_select(norm.running_mean.data, 0, index)
        new_norm.running_var.data = torch.index_select(norm.running_var.data, 0, index)
    return new_norm
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
    super_module = model
    for name in name_list[:-1]:
        super_module = getattr(super_module, name)
    leaf_module = getattr(super_module, name_list[-1])
    return super_module, leaf_module
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
