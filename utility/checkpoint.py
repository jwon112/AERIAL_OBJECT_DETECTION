import torch

def save_checkpoint(model, optimizer, epoch, best_metric, path, extra: dict = None):
    """
    모델 체크포인트 저장 함수
    Args:
        model: nn.Module 또는 state_dict
        optimizer: 옵티마이저 객체
        epoch: 현재 에포크
        best_metric: 최고 성능 지표 (예: best mAP)
        path: 저장 경로
        extra: 추가로 저장할 dict (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'best_metric': best_metric,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path, map_location=None):
    """
    모델 체크포인트 로드 함수
    Args:
        model: nn.Module
        optimizer: 옵티마이저 객체 (optional)
        path: 체크포인트 파일 경로
        map_location: 로드할 디바이스
    Returns:
        epoch, best_metric, checkpoint(dict)
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', None)
    return epoch, best_metric, checkpoint 