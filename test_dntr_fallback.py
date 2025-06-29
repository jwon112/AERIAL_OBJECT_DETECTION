#!/usr/bin/env python3
"""
DNTR Unified 모델 Fallback 테스트 스크립트
"""

import registry

def test_dntr_fallback():
    """DNTR unified 모델 fallback 테스트"""
    print("=== DNTR Unified Fallback 테스트 ===")
    
    # 모델 빌드 테스트
    try:
        ex_dict = {
            'Number of Classes': 10,
            'Batch Size': 16,
            'Image Size': 640,
            'Device': 'cpu'
        }
        model = registry.get_model('DNTR_UNIFIED', ex_dict)
        print(f"✅ DNTR_UNIFIED 모델 빌드 성공:")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Device: {getattr(model, 'device', 'N/A')}")
        
        # Forward 테스트
        import torch
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   - Forward pass 성공: {type(output)}")
        if isinstance(output, dict):
            print(f"   - Output keys: {list(output.keys())}")
        
    except Exception as e:
        print(f"❌ DNTR_UNIFIED 모델 빌드 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dntr_fallback() 