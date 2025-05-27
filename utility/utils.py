import os
import yaml
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from ultralytics import settings, YOLO
settings.update({'datasets_dir': './'})

def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
        torch.backends.cudnn.benchmark = False


def update_dataset_paths(dataset_root, dataset_name, iteration):
    
    dataset_dir = os.path.join(dataset_root, dataset_name)
    yaml_file = os.path.join(dataset_dir, f'data_iter_{iteration:02d}.yaml')
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'path' in data and data['path']:
        return True
    
    data['path'] = dataset_dir

    train_file = data.get('train')
    val_file = data.get('val')
    test_file = data.get('test')
    
    data['train'] = f'train_iter_{iteration:02d}.txt'
    data['val'] = f'val_iter_{iteration:02d}.txt'
    data['test'] = f'test_iter_{iteration:02d}.txt'
    
    with open(yaml_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    train_txt_path = os.path.join(dataset_dir, f'train_iter_{iteration:02d}.txt')
    val_txt_path = os.path.join(dataset_dir, f'val_iter_{iteration:02d}.txt')
    test_txt_path = os.path.join(dataset_dir, f'test_iter_{iteration:02d}.txt')
    
    for txt_path in [train_txt_path, val_txt_path, test_txt_path]:
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    if '/' in line or '\\' in line:
                        filename = os.path.basename(line)
                        dirname = os.path.dirname(line)
                        if dirname.startswith('images'):
                            updated_line = os.path.join(dataset_root, dataset_name, line)
                        else:
                            updated_line = os.path.join(dataset_root, dataset_name, 'images', filename)
                    else:
                        updated_line = os.path.join(dataset_root, dataset_name, 'images', line)
                    
                    updated_lines.append(updated_line + '\n')
            
            with open(txt_path, 'w') as f:
                f.writelines(updated_lines)
            
            print(f"{txt_path} 파일의 경로 업데이트 완료 ({len(lines)} 항목)")
        else:
            print(f"경고: {txt_path} 파일이 존재하지 않습니다.")
    
    return True


""" def train_model(ex_dict):
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    name = "Train"
    poject = f"{ex_dict['Output Dir']}/{ex_dict['Experiment Time']}/{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    ex_dict['Train Results'] = ex_dict['Model'].train(
        model = f"{ex_dict['Model Name']}.yaml",
        name=name,
        data=ex_dict['Data Config'] ,
        epochs=ex_dict['Epochs'],
        imgsz=ex_dict['Image Size'],
        batch=ex_dict['Batch Size'],
        patience=ex_dict['Early Stop'],
        save=True,
        device=ex_dict['Device'],
        exist_ok=True,
        verbose=False,
        optimizer=ex_dict['Optimizer'],
        lr0=ex_dict['LR'],  
        weight_decay = ex_dict['Weight Decay'],
        momentum = ex_dict['Momentum'],
        pretrained=False,
        amp=False,
        project = poject,
    )
    pt_path = f"{poject}/{name}/weights/best.pt"
    ex_dict['PT path'] = pt_path
    ex_dict['Model'].load(pt_path)
    return ex_dict
def evaluate_model(ex_dict):
    name = "Test"
    ex_dict['Test Results'] = ex_dict['Model'].val(data=ex_dict['Data Config'], 
                                                   name = name,
                                                   split='test', save=True)
    return ex_dict """
    
def format_measures(results, main_decimals=4, class_decimals=3, speed_decimals=1):
    # 결과가 딕셔너리인 경우 (YOLOoW_CLI) 또는 DetMetrics 객체인 경우 (YOLOv8) 모두 처리
    try:
        eval_dict = {
            'mAP@0.5': round(getattr(results.box, 'map50', 0.0), main_decimals),  
            'mAP@0.5:0.95': round(getattr(results.box, 'map', 0.0), main_decimals),  
            'Mean Precision': round(getattr(results.box, 'mp', 0.0), main_decimals),  
            'Mean Recall': round(getattr(results.box, 'mr', 0.0), main_decimals),  
            'mAP@0.75': round(getattr(results.box, 'map75', 0.0), main_decimals),  
        }
        
        if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
            for i, class_idx in enumerate(results.box.ap_class_index):
                if hasattr(results.box, 'names') and results.box.names is not None:
                    class_name = results.box.names[int(class_idx)]
                else:
                    class_name = f"Class_{int(class_idx)}"
                    
                # YOLOv8 방식 시도
                try:
                    p, r, ap50, ap = results.box.class_result(i)
                    eval_dict[f'{class_name}/Precision'] = round(p, class_decimals)
                    eval_dict[f'{class_name}/Recall'] = round(r, class_decimals)
                    eval_dict[f'{class_name}/mAP@0.5'] = round(ap50, class_decimals)
                    eval_dict[f'{class_name}/mAP@0.5:0.95'] = round(ap, class_decimals)
                except (AttributeError, TypeError):
                    # 메서드가 없는 경우 0으로 설정
                    eval_dict[f'{class_name}/Precision'] = 0.0
                    eval_dict[f'{class_name}/Recall'] = 0.0
                    eval_dict[f'{class_name}/mAP@0.5'] = 0.0
                    eval_dict[f'{class_name}/mAP@0.5:0.95'] = 0.0
        
        if hasattr(results, 'speed'):
            for k, v in results.speed.items():
                eval_dict[f'Speed/{k} (ms)'] = round(v, speed_decimals)
        
        return eval_dict
    
    except AttributeError:
        # 기본 메트릭 반환 (YOLOoW_CLI 또는 다른 구현의 경우)
        return {
            'mAP@0.5': 0.0,  
            'mAP@0.5:0.95': 0.0,  
            'Mean Precision': 0.0,  
            'Mean Recall': 0.0,  
            'mAP@0.75': 0.0,
        }
def merge_and_update_df(ex_dict, eval_dict, csv_path=None, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
    
    combined_dict = {**ex_dict}
    for k, v in eval_dict.items():
        if k not in exclude_columns:
            combined_dict[k] = v
    
    filtered_dict = {k: v for k, v in combined_dict.items() if k not in exclude_columns}
    
    new_row_df = pd.DataFrame([filtered_dict])
    
    existing_df = None
    existing_columns = []
    
    if csv_path and os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            existing_columns = list(existing_df.columns)
        except Exception as e:
            existing_df = None
    
    priority_columns = ['Experiment Time', 'Train Time', 'Iteration', 'Dataset Name', 'Model Name']
    new_columns = list(new_row_df.columns)
    eval_columns = list(eval_dict.keys())
    
    if existing_columns:
        ordered_columns = [col for col in priority_columns if col in new_columns or col in existing_columns]
        ordered_columns += [col for col in eval_columns if (col in new_columns or col in existing_columns) and col not in ordered_columns]
        ordered_columns += [col for col in existing_columns if col not in ordered_columns]
        ordered_columns += [col for col in new_columns if col not in ordered_columns]
    else:
        ordered_columns = [col for col in priority_columns if col in new_columns]
        ordered_columns += [col for col in eval_columns if col in new_columns and col not in ordered_columns]
        ordered_columns += [col for col in new_columns if col not in ordered_columns]
    
    if existing_df is not None:
        for col in new_columns:
            if col not in existing_df.columns:
                existing_df[col] = None
        
        for col in existing_columns:
            if col not in new_row_df.columns:
                new_row_df[col] = None
        
        all_columns = ordered_columns
        existing_df = existing_df.reindex(columns=all_columns)
        new_row_df = new_row_df.reindex(columns=all_columns)
        
        df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        df = new_row_df
    
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"DataFrame이 '{csv_path}'에 저장되었습니다.")
    return df