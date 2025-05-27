from Models.YoloOW.yoloow_utils import build_yoloow_model, train_yoloow_model, eval_yoloow_model, test_yoloow_model
from Models.YoloOW.yoloow_cli import build_yoloow_model_cli, train_yoloow_model_cli, eval_yoloow_model_cli, test_yoloow_model_cli
from Models.YOLOH.yoloh_utils import build_yoloh_model, train_yoloh_model, eval_yoloh_model, test_yoloh_model
from functools import partial
from Models.YOLOH.config.yoloh_config import yoloh_config
from Models.ultralytics.yolov8_utils import (
    build_yolov8_model, train_yolov8_model, eval_yolov8_model, test_yolov8_model
)
from Models.YOLOH.yoloh_cli import build_yoloh_model_cli, train_yoloh_model_cli, eval_yoloh_model_cli, test_yoloh_model_cli

model_registry = {
    'YoloOW': {
        'build': partial(build_yoloow_model, cfg='yoloOW.yaml'), 
        'train': train_yoloow_model,
        'eval': eval_yoloow_model,
        'test': test_yoloow_model
    },
    'YoloOW_CLI': {
        'build': partial(build_yoloow_model_cli, cfg='yoloOW.yaml'), 
        'train': train_yoloow_model_cli,
        'eval': eval_yoloow_model_cli,
        'test': test_yoloow_model_cli
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
    },
    'yolov8n': {
        'build': partial(build_yolov8_model, cfg='Models/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml'),
        'train': train_yolov8_model,
        'eval': eval_yolov8_model,
        'test': test_yolov8_model
    },
    'YOLOH_CLI': {
        'build': partial(build_yoloh_model_cli, cfg=None),
        'train': train_yoloh_model_cli,
        'eval': eval_yoloh_model_cli,
        'test': test_yoloh_model_cli
    },
}

def get_model(model_name, ex_dict):
    if model_name not in model_registry:
        raise ValueError(f"Unsupported model: {model_name}")
    return model_registry[model_name]['build'](ex_dict=ex_dict)

def get_pipeline(model_name):
    return model_registry[model_name]['train'], model_registry[model_name]['eval'], model_registry[model_name]['test']
