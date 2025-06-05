#!/usr/bin/env python3
"""
DEBUG 로그 파일 관리 유틸리티
"""

import os
import logging
import time
from datetime import datetime
from typing import Optional

class DebugLogger:
    """DEBUG 메시지를 별도 파일로 저장하는 로거"""
    
    def __init__(self, experiment_id: str = None, log_dir: str = "logs"):
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.log_dir = log_dir
        self.log_file = self._setup_log_file()
        self.logger = self._setup_logger()
        
    def _generate_experiment_id(self) -> str:
        """실험 ID 생성"""
        return datetime.now().strftime("%y%m%d_%H%M%S")
    
    def _setup_log_file(self) -> str:
        """로그 파일 경로 설정"""
        os.makedirs(self.log_dir, exist_ok=True)
        return os.path.join(self.log_dir, f"debug_{self.experiment_id}.log")
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"debug_{self.experiment_id}")
        logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.propagate = False  # 상위 로거로 전파 방지
        
        return logger
    
    def debug(self, message: str, show_console: bool = False):
        """DEBUG 메시지 로깅"""
        self.logger.debug(message)
        if show_console:
            print(f"[DEBUG] {message}")
    
    def info(self, message: str, show_console: bool = True):
        """INFO 메시지 로깅"""
        self.logger.info(message)
        if show_console:
            print(f"[INFO] {message}")
    
    def get_log_file_path(self) -> str:
        """로그 파일 경로 반환"""
        return self.log_file

# 전역 디버그 로거 인스턴스
_global_debug_logger: Optional[DebugLogger] = None

def init_debug_logger(experiment_id: str = None) -> DebugLogger:
    """전역 디버그 로거 초기화"""
    global _global_debug_logger
    _global_debug_logger = DebugLogger(experiment_id)
    print(f"📋 DEBUG 로그 파일: {_global_debug_logger.get_log_file_path()}")
    return _global_debug_logger

def get_debug_logger() -> Optional[DebugLogger]:
    """전역 디버그 로거 반환"""
    return _global_debug_logger

def debug_log(message: str, show_console: bool = False):
    """편의 함수: DEBUG 로그 출력"""
    if _global_debug_logger:
        _global_debug_logger.debug(message, show_console)
    elif show_console:
        print(f"[DEBUG] {message}")

def info_log(message: str, show_console: bool = True):
    """편의 함수: INFO 로그 출력"""
    if _global_debug_logger:
        _global_debug_logger.info(message, show_console)
    elif show_console:
        print(f"[INFO] {message}") 