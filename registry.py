from Models.YoloOW.yoloow_utils import build_yoloow_model, train_yoloow_model, eval_yoloow_model, test_yoloow_model
from Models.YOLOH.yoloh_utils import build_yoloh_model, train_yoloh_model, eval_yoloh_model, test_yoloh_model
from functools import partial
from Models.YOLOH.config.yoloh_config import yoloh_config

model_registry = {
    'YoloOW': {
        'build': partial(build_yoloow_model, cfg='yoloOW.yaml'), 
        'train': train_yoloow_model,
        'eval': eval_yoloow_model,
        'test': test_yoloow_model
    },
    'yoloh18': {
        'build': partial(build_yoloh_model, cfg=yoloh_config['yoloh18']),
        'train': train_yoloh_model,
        'eval': eval_yoloh_model,
        'test' : test_yoloh_model
    },
    'yoloh50' : {
        'build': partial(build_yoloh_model, cfg=yoloh_config['yoloh50']),
        'train': train_yoloh_model,
        'eval': eval_yoloh_model,
        'test' : test_yoloh_model

    },
    'yoloh101' : {
        'build': partial(build_yoloh_model, cfg=yoloh_config['yoloh101']),
        'train': train_yoloh_model,
        'eval': eval_yoloh_model,
        'test' : test_yoloh_model
    }
}

def get_model(model_name, ex_dict):
    if model_name not in model_registry:
        raise ValueError(f"Unsupported model: {model_name}")
    return model_registry[model_name]['build'](ex_dict=ex_dict)

def get_pipeline(model_name):
    return model_registry[model_name]['train'], model_registry[model_name]['eval'], model_registry[model_name]['test']
