import json


def flatten_dict(nested_dict):

    flattened = {}
    for section, params in nested_dict.items():
        for param_name, param_value in params.items():
            flattened[param_name] = param_value
    return flattened

def load_hyperparameters(json_path, args=None):

    with open(json_path, 'r') as f:
        config = json.load(f)
    
    flattened_config = flatten_dict(config)
    
    if args is not None:
        if hasattr(args, 'retrain_lr'):
            flattened_config['retrain_lr'] = args.retrain_lr
            flattened_config['finetune_lr'] = args.retrain_lr


    
    globals().update(flattened_config)
    
    return flattened_config