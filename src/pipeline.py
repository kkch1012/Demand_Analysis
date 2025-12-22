"""
자동화 파이프라인 모듈
- 데이터 로딩부터 예측까지 자동화
- 새 데이터가 들어오면 자동 재학습
"""

import numpy as np
import pandas as pd
import yaml
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import SalesPredictionModel


class AutomatedPipeline:
    """자동화된 매출 예측 파이프라인"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 경로 설정
        self.input_data_path = self.config['data']['input_data_path']
        self.sales_data_path = self.config['data']['sales_data_path']
        self.model_save_path = self.config['data']['model_save_path']
        self.prediction_save_path = self.config['data']['prediction_save_path']
        
        # 컴포넌트 초기화
        self.data_loader = DataLoader(self.input_data_path, self.sales_data_path)
        self.feature_engineer = FeatureEngineer()
        self.model = None
        
        # 경로 생성
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.prediction_save_path, exist_ok=True)
    
    def check_new_data(self) -> bool:
        """새 데이터가 있는지 확인"""
        info = self.data_loader.get_latest_data_info()
        
        min_months = self.config['automation']['min_data_months']
        total_months = max(
            len(info['input_data_months']),
            info['total_sales_months']
        )
        
        return total_months >= min_months
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터 준비 및 전처리
        
        Returns:
            (X_time_train, X_word_train, y_train, data_info)
        """
        # 데이터 로딩
        print("데이터 로딩 중...")
        input_data = self.data_loader.load_input_data()
        sales_data = self.data_loader.load_sales_data()
        
        if sales_data.empty:
            raise ValueError("매출 데이터가 없습니다.")
        
        # 데이터 결합
        print("데이터 결합 중...")
        combined_data = self.data_loader.combine_data(input_data, sales_data)
        
        # 시퀀스 준비
        sequence_length = self.config['model']['time_series']['sequence_length']
        print(f"시퀀스 데이터 준비 중 (길이: {sequence_length})...")
        X_sequences, y = self.data_loader.prepare_sequences(
            combined_data, 
            sequence_length
        )
        
        if len(X_sequences) == 0:
            raise ValueError(f"시퀀스 데이터가 부족합니다. 최소 {sequence_length + 1}개월의 데이터가 필요합니다.")
        
        # 특성 분리
        # 시계열 특성: 시퀀스의 모든 특성
        # 단어 특성: 마지막 시퀀스의 단어 점수들
        n_samples, n_timesteps, n_features = X_sequences.shape
        
        # 단어 컬럼 인덱스 찾기
        word_columns = [col for col in combined_data.columns if col.startswith('word_score_')]
        
        # 시계열 입력: 전체 시퀀스
        X_time = X_sequences.copy()
        
        # 단어 입력: 마지막 시퀀스의 단어 점수들만
        # 시퀀스의 마지막 타임스텝에서 단어 점수 추출
        if word_columns:
            # combined_data에서 단어 컬럼 인덱스 찾기
            all_columns = list(combined_data.columns)
            word_indices = [all_columns.index(col) for col in word_columns if col in all_columns]
            
            # 시퀀스에서 단어 특성 추출
            X_word = X_sequences[:, -1, word_indices] if word_indices else np.zeros((n_samples, len(word_columns)))
        else:
            X_word = np.zeros((n_samples, 1))
        
        # 데이터 정보
        data_info = {
            'n_samples': n_samples,
            'n_timesteps': n_timesteps,
            'n_time_features': n_features,
            'n_word_features': X_word.shape[1] if len(X_word.shape) > 1 else 1,
            'word_columns': word_columns
        }
        
        return X_time, X_word, y, data_info
    
    def train_model(self, 
                   X_time: np.ndarray,
                   X_word: np.ndarray,
                   y: np.ndarray,
                   data_info: Dict,
                   retrain: bool = False) -> SalesPredictionModel:
        """
        모델 학습
        
        Args:
            X_time: 시계열 입력
            X_word: 단어 특성 입력
            y: 타겟
            data_info: 데이터 정보
            retrain: 재학습 여부
            
        Returns:
            학습된 모델
        """
        # 특성 정규화
        print("특성 정규화 중...")
        X_time_norm, _ = self.feature_engineer.normalize_features(X_time)
        X_word_norm, _ = self.feature_engineer.normalize_features(X_word)
        y_norm, _ = self.feature_engineer.normalize_sales(y)
        
        # 모델 구축
        print("모델 구축 중...")
        model = SalesPredictionModel(self.config['model'])
        model.build_model(
            sequence_length=data_info['n_timesteps'],
            n_time_features=data_info['n_time_features'],
            n_word_features=data_info['n_word_features']
        )
        
        # 학습/검증 데이터 분할
        val_split = self.config['model']['training']['validation_split']
        split_idx = int(len(X_time_norm) * (1 - val_split))
        
        X_time_train = X_time_norm[:split_idx]
        X_time_val = X_time_norm[split_idx:]
        X_word_train = X_word_norm[:split_idx]
        X_word_val = X_word_norm[split_idx:]
        y_train = y_norm[:split_idx]
        y_val = y_norm[split_idx:]
        
        validation_data = ([X_time_val, X_word_val], y_val) if len(self.config['model']['time_series']['lstm_units']) > 0 else (X_word_val, y_val)
        
        # 학습
        print("모델 학습 중...")
        model.train(
            X_time_train,
            X_word_train,
            y_train,
            validation_data=validation_data,
            verbose=1
        )
        
        self.model = model
        self.feature_engineer = self.feature_engineer  # 스케일러 저장
        
        return model
    
    def predict_next_month(self,
                          X_time: np.ndarray,
                          X_word: np.ndarray) -> float:
        """
        다음 달 매출 예측
        
        Args:
            X_time: 시계열 입력 (마지막 시퀀스)
            X_word: 단어 특성 입력 (다음 달 단어 점수)
            
        Returns:
            예측된 매출
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train_model()을 실행하세요.")
        
        # 정규화
        X_time_norm = self.feature_engineer.scaler.transform(
            X_time.reshape(-1, X_time.shape[-1])
        ).reshape(X_time.shape)
        X_word_norm = self.feature_engineer.scaler.transform(X_word.reshape(-1, X_word.shape[-1])).reshape(X_word.shape) if X_word.ndim > 1 else self.feature_engineer.scaler.transform(X_word.reshape(-1, 1)).flatten()
        
        # 예측
        prediction_norm = self.model.predict(X_time_norm, X_word_norm)
        
        # 역정규화
        prediction = self.feature_engineer.denormalize_sales(prediction_norm)
        
        return prediction[0] if len(prediction) > 0 else 0.0
    
    def save_model(self, model_name: Optional[str] = None):
        """모델 저장"""
        if model_name is None:
            model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        
        filepath = os.path.join(self.model_save_path, model_name)
        self.model.save(filepath)
        
        # 스케일러도 저장 (joblib 사용)
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        joblib.dump(self.feature_engineer, scaler_path)
        
        print(f"모델 저장 완료: {filepath}")
        return filepath
    
    def load_latest_model(self) -> bool:
        """최신 모델 로드"""
        model_files = list(Path(self.model_save_path).glob("*.h5"))
        if not model_files:
            return False
        
        latest_model = max(model_files, key=os.path.getctime)
        
        self.model = SalesPredictionModel(self.config['model'])
        self.model.load(str(latest_model))
        
        # 스케일러 로드
        scaler_path = str(latest_model).replace('.h5', '_scaler.pkl')
        if os.path.exists(scaler_path):
            import joblib
            self.feature_engineer = joblib.load(scaler_path)
        
        print(f"모델 로드 완료: {latest_model}")
        return True
    
    def run_full_pipeline(self, retrain: bool = False) -> Dict:
        """
        전체 파이프라인 실행
        
        Args:
            retrain: 강제 재학습 여부
            
        Returns:
            실행 결과 딕셔너리
        """
        print("=" * 50)
        print("자동화 파이프라인 시작")
        print("=" * 50)
        
        # 데이터 확인
        if not self.check_new_data():
            print("경고: 충분한 데이터가 없습니다.")
            return {'status': 'insufficient_data'}
        
        # 데이터 준비
        try:
            X_time, X_word, y, data_info = self.prepare_data()
        except Exception as e:
            print(f"데이터 준비 실패: {e}")
            return {'status': 'error', 'message': str(e)}
        
        # 모델 학습 또는 로드
        if retrain or not self.load_latest_model():
            print("새 모델 학습 시작...")
            self.train_model(X_time, X_word, y, data_info, retrain=retrain)
            self.save_model()
        else:
            print("기존 모델 사용")
        
        # 다음 달 예측
        # 다음 달의 단어-점수 데이터를 로드하여 예측
        if len(X_time) > 0:
            last_sequence = X_time[-1:].copy()
            
            # 다음 달의 단어-점수 데이터 로드
            next_month_word_features = self._load_next_month_word_features(data_info['word_columns'])
            
            if next_month_word_features is None:
                print("경고: 다음 달 단어-점수 데이터가 없습니다. 마지막 달 데이터를 사용합니다.")
                last_word_features = X_word[-1:].copy()
            else:
                last_word_features = next_month_word_features
            
            prediction = self.predict_next_month(last_sequence, last_word_features)
            
            # 결과 저장
            result = {
                'status': 'success',
                'prediction': float(prediction),
                'data_info': data_info,
                'timestamp': datetime.now().isoformat()
            }
            
            # 예측 결과 저장
            prediction_file = os.path.join(
                self.prediction_save_path,
                f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(prediction_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n다음 달 예측 매출: {prediction:,.2f}")
            print(f"결과 저장: {prediction_file}")
            
            return result
        
        return {'status': 'no_data'}
    
    def _load_next_month_word_features(self, word_columns: List[str]) -> Optional[np.ndarray]:
        """
        다음 달의 단어-점수 데이터 로드
        
        Args:
            word_columns: 단어 컬럼명 리스트
            
        Returns:
            다음 달 단어 특성 배열 또는 None
        """
        from datetime import datetime, timedelta
        import calendar
        
        # 다음 달 계산
        today = datetime.now()
        if today.month == 12:
            next_month = datetime(today.year + 1, 1, 1)
        else:
            next_month = datetime(today.year, today.month + 1, 1)
        
        next_month_str = next_month.strftime('%Y-%m')
        
        # 다음 달 단어-점수 데이터 로드
        input_data = self.data_loader.load_input_data(month=next_month_str)
        
        if not input_data or next_month_str not in input_data:
            return None
        
        # 단어-점수 데이터를 벡터로 변환
        word_scores = input_data[next_month_str]
        feature_vector = []
        
        for word_col in word_columns:
            # 컬럼명에서 단어 추출 (word_score_제거)
            word = word_col.replace('word_score_', '')
            score = word_scores.get(word, 0.0)
            feature_vector.append(score)
        
        return np.array([feature_vector]) if feature_vector else None
