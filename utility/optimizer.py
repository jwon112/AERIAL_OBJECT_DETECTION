from torch import optim


def build_optimizer(model,
                    base_lr=0.0,
                    backbone_lr=0.0,
                    name='sgd',
                    momentum=0.,
                    weight_decay=0.):
    name = name.lower()
    print('==============================')
    print('Optimizer: {}'.format(name))
    print('--momentum: {}'.format(momentum))
    print('--weight_decay: {}'.format(weight_decay))

    if base_lr == backbone_lr:
        param_dicts = model.parameters()
    else:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": backbone_lr,
            },
        ]

    if name == 'sgd':
        optimizer = optim.SGD(param_dicts, 
                                lr=base_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    elif name == 'adam':
        optimizer = optim.Adam(param_dicts, 
                                lr=base_lr,
                                weight_decay=weight_decay)
                                
    elif name == 'adamw':
        optimizer = optim.AdamW(param_dicts, 
                                lr=base_lr,
                                weight_decay=weight_decay)
                                
    return optimizer
