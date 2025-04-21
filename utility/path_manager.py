# utility/path_manager.py

# Models 에 모델 소스 코드를 넣을 경우 기존 모델 소스 코드의 경우 각자 별도의 entry point를 간주하기 때문에
# 서로 다른 entry point로 인한 경로 충돌 문제가 있음으로 해당 코드를 사용하여 경로를 관리합니다.

from contextlib import contextmanager
import sys, pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODEL_ROOTS = {
    "YOLOH": _PROJECT_ROOT / "Models" / "YOLOH",
    "YoloOW": _PROJECT_ROOT / "Models" / "YoloOW",
    #여기에 아무 모델이든 추가 가능
}

@contextmanager
def use_model_root(model_key: str, verbose=False):
    root = _MODEL_ROOTS[model_key]
    # 👉 1) 현재 'models' 패키지 백업
    backup_models = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith("models.")}
    conflict_keys = [k for k in sys.modules if k == "utils" or k.startswith("utils.")]
    #conflict_keys = [k for k in sys.modules if any(k.startswith(prefix) for prefix in ["utils", "loss", "dataloader"])]
    cached_utils = {k: sys.modules.pop(k) for k in conflict_keys if k in sys.modules}

    # 👉 2) 캐시 제거
    for k in list(backup_models):
        sys.modules.pop(k, None)

    # 👉 3) 경로 삽입
    sys.path.insert(0, str(root))
    if verbose:
        print(f"[path_manager] + {root}")

    try:
        yield
    finally:
        # 👉 4) 경로 복구
        if str(root) in sys.path:
            sys.path.remove(str(root))
            if verbose:
                print(f"[path_manager] - {root}")
        # 👉 5) 원래 'models' 캐시 복원
        sys.modules.update(backup_models)
        sys.modules.update(cached_utils)
