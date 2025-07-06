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
    # MSNet 평가 결과 딕셔너리인 경우 처리
    if isinstance(results, dict):
        eval_dict = {}
        
        # MSNet 결과 매핑
        if 'mAP' in results:
            eval_dict['mAP@0.5:0.95'] = round(results['mAP'], main_decimals)
        if 'AP50' in results:
            eval_dict['mAP@0.5'] = round(results['AP50'], main_decimals)
        if 'AP75' in results:
            eval_dict['mAP@0.75'] = round(results['AP75'], main_decimals)
        if 'APs' in results:
            eval_dict['mAP@small'] = round(results['APs'], main_decimals)
        if 'APm' in results:
            eval_dict['mAP@medium'] = round(results['APm'], main_decimals)
        if 'APl' in results:
            eval_dict['mAP@large'] = round(results['APl'], main_decimals)
        
        # 기본값 설정 (없는 경우)
        eval_dict.setdefault('mAP@0.5', 0.0)
        eval_dict.setdefault('mAP@0.5:0.95', 0.0)
        eval_dict.setdefault('Mean Precision', 0.0)
        eval_dict.setdefault('Mean Recall', 0.0)
        eval_dict.setdefault('mAP@0.75', 0.0)
        
        return eval_dict
    
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

def save_evaluation_results(results: dict, output_path: str, model_name: str, dataset_name: str, 
                           experiment_id: str = None, format_type: str = 'json'):
    """
    평가 결과를 파일로 저장
    
    Args:
        results: 평가 결과 딕셔너리
        output_path: 저장할 파일 경로
        model_name: 모델 이름
        dataset_name: 데이터셋 이름
        experiment_id: 실험 ID (선택사항)
        format_type: 저장 형식 ('json', 'csv', 'txt')
    """
    import json
    from pathlib import Path
    
    # 출력 디렉토리 생성
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 메타데이터 추가
    save_data = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    if experiment_id:
        save_data['experiment_id'] = experiment_id
    
    try:
        if format_type.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        elif format_type.lower() == 'csv':
            # 결과를 CSV 형식으로 변환
            import pandas as pd
            
            # 평면화된 결과 딕셔너리 생성
            flat_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_results[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_results[key] = value
            
            # 메타데이터 추가
            flat_results['model_name'] = model_name
            flat_results['dataset_name'] = dataset_name
            flat_results['timestamp'] = save_data['timestamp']
            if experiment_id:
                flat_results['experiment_id'] = experiment_id
            
            df = pd.DataFrame([flat_results])
            df.to_csv(output_path, index=False)
        
        elif format_type.lower() == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Timestamp: {save_data['timestamp']}\n")
                if experiment_id:
                    f.write(f"Experiment ID: {experiment_id}\n")
                f.write("\nResults:\n")
                f.write("=" * 50 + "\n")
                
                for key, value in results.items():
                    if isinstance(value, dict):
                        f.write(f"\n{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
        
        print(f"평가 결과가 '{output_path}'에 저장되었습니다.")
        
    except Exception as e:
        print(f"평가 결과 저장 중 오류 발생: {e}")
        raise

def save_model_params_info(model, save_dir, model_name, config_info=None):
    """
    모델 파라미터 정보를 파일에 저장하는 공통 함수
    
    Args:
        model: PyTorch 모델 객체
        save_dir: 저장할 디렉토리
        model_name: 모델 이름
        config_info: 추가 설정 정보 (선택사항)
    """
    if not hasattr(model, 'parameters'):
        print(f"⚠️ {model_name}: PyTorch 모델이 아니므로 파라미터 정보를 계산할 수 없습니다.")
        return
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print(f"📊 {model_name} 모델 파라미터 정보:")
        print(f"  총 파라미터: {total_params:,}개")
        print(f"  학습 가능: {trainable_params:,}개")
        print(f"  고정: {non_trainable_params:,}개")
        
        # 파라미터 정보를 파일에 저장
        os.makedirs(save_dir, exist_ok=True)
        param_info_path = os.path.join(save_dir, 'model_params.txt')
        with open(param_info_path, 'w') as f:
            f.write(f"Total Parameters: {total_params}\n")
            f.write(f"Trainable Parameters: {trainable_params}\n")
            f.write(f"Non-trainable Parameters: {non_trainable_params}\n")
            f.write(f"Model Name: {model_name}\n")
            if config_info:
                f.write(f"Model Config: {config_info}\n")
        
        print(f"📁 모델 파라미터 정보 저장: {param_info_path}")
        
    except Exception as e:
        print(f"❌ {model_name} 파라미터 정보 저장 중 오류: {e}")

def load_model_params_info(save_dir):
    """
    저장된 모델 파라미터 정보를 로드하는 공통 함수
    
    Args:
        save_dir: 파라미터 정보 파일이 있는 디렉토리
        
    Returns:
        dict: 파라미터 정보 딕셔너리
    """
    print(f"[DEBUG] load_model_params_info 시작")
    print(f"[DEBUG] save_dir: {save_dir}")
    
    param_info = {}
    param_file_path = os.path.join(save_dir, 'model_params.txt')
    print(f"[DEBUG] 찾는 파일 경로: {param_file_path}")
    
    if not os.path.exists(param_file_path):
        print(f"[DEBUG] 파라미터 정보 파일이 존재하지 않습니다: {param_file_path}")
        return param_info
    
    print(f"[DEBUG] 파라미터 정보 파일 존재함: {param_file_path}")
    
    try:
        with open(param_file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"[DEBUG] 파일에서 읽은 라인 수: {len(lines)}")
        for i, line in enumerate(lines):
            print(f"[DEBUG] 라인 {i}: {repr(line)}")
            
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                print(f"[DEBUG] 파싱 중: key='{key}', value='{value}'")
                
                if key in ['Total Parameters', 'Trainable Parameters', 'Non-trainable Parameters']:
                    try:
                        param_info[key] = int(value)
                        print(f"[DEBUG] 정수로 저장: {key} = {param_info[key]}")
                    except ValueError as e:
                        print(f"[DEBUG] 정수 변환 실패: {key}={value}, 에러: {e}")
                        param_info[key] = value
                else:
                    param_info[key] = value
                    print(f"[DEBUG] 문자열로 저장: {key} = {value}")
                    
        print(f"[DEBUG] 최종 param_info: {param_info}")
        print(f"📊 모델 파라미터 정보 로드: {param_info}")
                    
    except Exception as e:
        print(f"[DEBUG] 모델 파라미터 파일 파싱 에러: {e}")
        import traceback
        traceback.print_exc()
    
    return param_info

def add_model_params_to_ex_dict(ex_dict, save_dir):
    """
    ex_dict에 모델 파라미터 정보를 추가하는 공통 함수
    
    Args:
        ex_dict: 실험 딕셔너리
        save_dir: 파라미터 정보 파일이 있는 디렉토리
    """
    print(f"[DEBUG] add_model_params_to_ex_dict 시작")
    print(f"[DEBUG] ex_dict 입력 키 개수: {len(ex_dict)}")
    print(f"[DEBUG] save_dir: {save_dir}")
    
    param_info = load_model_params_info(save_dir)
    print(f"[DEBUG] load_model_params_info 결과: {param_info}")
    
    if param_info:
        print(f"[DEBUG] param_info가 비어있지 않음, 항목 개수: {len(param_info)}")
        for key, value in param_info.items():
            print(f"[DEBUG] ex_dict에 추가: {key} = {value}")
            ex_dict[key] = value
        print(f"[DEBUG] 📊 모델 파라미터 정보가 ex_dict에 추가되었습니다.")
        print(f"[DEBUG] ex_dict 결과 키 개수: {len(ex_dict)}")
        param_keys = [k for k in ex_dict.keys() if 'Parameter' in k or 'Model' in k]
        print(f"[DEBUG] ex_dict에 있는 파라미터 관련 키들: {param_keys}")
    else:
        print(f"[DEBUG] param_info가 비어있음 - 파라미터 정보가 추가되지 않음")
    
    return ex_dict

def extract_params_from_checkpoint(checkpoint_path, model_name, save_dir):
    """
    PyTorch 체크포인트에서 모델 파라미터 정보를 추출하여 저장
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        model_name: 모델 이름
        save_dir: 저장할 디렉토리
    """
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
        return
    
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 체크포인트에서 state_dict 추출
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 파라미터 개수 계산
        total_params = 0
        trainable_params = 0
        
        for name, param in state_dict.items():
            if hasattr(param, 'numel'):
                param_count = param.numel()
            else:
                param_count = param.size().numel() if hasattr(param, 'size') else 0
            
            total_params += param_count
            # 대부분의 파라미터는 학습 가능하다고 가정
            trainable_params += param_count
        
        non_trainable_params = total_params - trainable_params
        
        print(f"📊 {model_name} 모델 파라미터 정보 (체크포인트에서 추출):")
        print(f"  총 파라미터: {total_params:,}개")
        print(f"  학습 가능: {trainable_params:,}개")
        print(f"  고정: {non_trainable_params:,}개")
        
        # 파라미터 정보를 파일에 저장
        os.makedirs(save_dir, exist_ok=True)
        param_info_path = os.path.join(save_dir, 'model_params.txt')
        with open(param_info_path, 'w') as f:
            f.write(f"Total Parameters: {total_params}\n")
            f.write(f"Trainable Parameters: {trainable_params}\n")
            f.write(f"Non-trainable Parameters: {non_trainable_params}\n")
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Checkpoint Path: {checkpoint_path}\n")
        
        print(f"📁 모델 파라미터 정보 저장: {param_info_path}")
        
    except Exception as e:
        print(f"❌ {model_name} 체크포인트에서 파라미터 정보 추출 중 오류: {e}")