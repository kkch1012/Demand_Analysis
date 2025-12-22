"""
Blinkit 수요 예측 파이프라인 모듈
- 데이터 로딩부터 예측까지 자동화
- 학습, 평가, 예측 통합
"""

import numpy as np
import pandas as pd
import yaml
import os
import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.blinkit.data_loader import BlinkitDataLoader
from src.blinkit.feature_engineering import BlinkitFeatureEngineer
from src.blinkit.model import BlinkitDemandModel


class BlinkitPipeline:
    """Blinkit 수요 예측 파이프라인"""
    
    def __init__(self, config_path: str = "config/blinkit_config.yaml", freq: str = None):
        """
        Args:
            config_path: 설정 파일 경로
            freq: 집계 주기 ('daily', 'weekly', 'monthly') - None이면 config에서 읽음
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 경로 설정
        self.raw_data_path = self.config['data']['raw_data_path']
        self.processed_data_path = self.config['data']['processed_data_path']
        self.model_save_path = self.config['data']['model_save_path']
        self.prediction_save_path = self.config['data']['prediction_save_path']
        
        # 예측 설정
        self.target_col = self.config['prediction']['target']
        self.sequence_length = self.config['model']['time_series']['sequence_length']
        
        # 집계 주기 설정
        self.freq = freq if freq else self.config['prediction'].get('aggregation', 'monthly')
        
        # 컴포넌트 초기화
        self.data_loader = BlinkitDataLoader(self.raw_data_path)
        self.feature_engineer = BlinkitFeatureEngineer()
        self.model = None
        
        # 데이터 저장
        self.agg_data = None  # monthly_data 대신 agg_data 사용
        self.X_seq = None
        self.X_extra = None
        self.y = None
        
        # 경로 생성
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.prediction_save_path, exist_ok=True)
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        데이터 로딩 및 전처리
        
        Returns:
            전처리된 데이터프레임
        """
        freq_name = {'daily': '일간', 'weekly': '주간', 'monthly': '월간'}[self.freq]
        
        print("=" * 60)
        print(f"1. 데이터 로딩 및 전처리 ({freq_name} 집계)")
        print("=" * 60)
        
        # 원본 데이터 로딩
        self.data_loader.load_raw_data()
        
        # 집계 데이터 생성
        self.agg_data = self.data_loader.prepare_full_dataset(freq=self.freq)
        
        # 시간 특성 추가 (date 컬럼 사용)
        if 'date' in self.agg_data.columns:
            self.agg_data['month'] = self.agg_data['date']  # 호환성
        
        # 지연 특성 추가 (주기에 따라 lag 조절)
        if self.freq == 'daily':
            lags = [1, 7, 14, 30]  # 1일, 1주, 2주, 1달
        elif self.freq == 'weekly':
            lags = [1, 2, 4, 8]   # 1주, 2주, 1달, 2달
        else:
            lags = [1, 2, 3]      # 1달, 2달, 3달
            
        self.agg_data = self.feature_engineer.add_lag_features(
            self.agg_data, 
            self.target_col,
            lags=lags
        )
        
        # 이동평균 특성 추가
        if self.freq == 'daily':
            windows = [7, 14, 30]
        elif self.freq == 'weekly':
            windows = [2, 4, 8]
        else:
            windows = [2, 3]
            
        self.agg_data = self.feature_engineer.add_rolling_features(
            self.agg_data,
            self.target_col,
            windows=windows
        )
        
        # 성장률 특성 추가
        self.agg_data = self.feature_engineer.add_growth_features(
            self.agg_data,
            self.target_col
        )
        
        # 전처리된 데이터 저장
        processed_file = os.path.join(
            self.processed_data_path, 
            f'blinkit_{self.freq}_processed.csv'
        )
        self.agg_data.to_csv(processed_file, index=False)
        print(f"\n전처리된 데이터 저장: {processed_file}")
        
        # 하위 호환성
        self.monthly_data = self.agg_data
        
        return self.agg_data
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        학습용 데이터 준비
        
        Returns:
            (X_seq, X_extra, y) - 시퀀스, 추가 특성, 타겟
        """
        print("\n" + "=" * 60)
        print("2. 학습 데이터 준비")
        print("=" * 60)
        
        if self.monthly_data is None:
            self.load_and_prepare_data()
        
        # 시퀀스 데이터 준비
        self.X_seq, self.X_extra, self.y = self.data_loader.prepare_sequences(
            self.monthly_data,
            self.target_col,
            self.sequence_length
        )
        
        return self.X_seq, self.X_extra, self.y
    
    def train_model(self, 
                   X_seq: np.ndarray = None,
                   X_extra: np.ndarray = None,
                   y: np.ndarray = None,
                   verbose: int = 1) -> Dict:
        """
        모델 학습
        
        Args:
            X_seq: 시퀀스 데이터
            X_extra: 추가 특성
            y: 타겟
            verbose: 출력 상세도
            
        Returns:
            학습 결과 딕셔너리
        """
        print("\n" + "=" * 60)
        print("3. 모델 학습")
        print("=" * 60)
        
        if X_seq is None:
            X_seq, X_extra, y = self.X_seq, self.X_extra, self.y
        
        if X_seq is None:
            raise ValueError("학습 데이터가 없습니다. prepare_training_data()를 먼저 실행하세요.")
        
        # 데이터 정규화
        print("\n특성 정규화 중...")
        X_seq_norm, _ = self.feature_engineer.normalize_sequences(X_seq)
        X_extra_norm, _ = self.feature_engineer.normalize_features(X_extra)
        y_norm, _ = self.feature_engineer.normalize_target(y)
        
        # 모델 구축
        print("\n모델 구축 중...")
        self.model = BlinkitDemandModel(self.config['model'])
        self.model.build_model(
            sequence_length=X_seq.shape[1],
            n_sequence_features=X_seq.shape[2],
            n_extra_features=X_extra.shape[1]
        )
        
        # 모델 요약
        self.model.summary()
        
        # 학습
        print("\n학습 시작...")
        validation_split = self.config['model']['training']['validation_split']
        history = self.model.train(
            X_seq_norm,
            X_extra_norm,
            y_norm,
            validation_split=validation_split,
            verbose=verbose
        )
        
        # 결과 정리
        result = {
            'epochs_trained': len(history.history['loss']),
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history.get('val_loss', [None])[-1],
            'final_mae': history.history['mae'][-1],
            'final_val_mae': history.history.get('val_mae', [None])[-1]
        }
        
        print(f"\n학습 완료!")
        print(f"  - Epochs: {result['epochs_trained']}")
        print(f"  - Final Loss: {result['final_loss']:.4f}")
        print(f"  - Final Val Loss: {result['final_val_loss']:.4f}")
        
        return result
    
    def evaluate_model(self) -> Dict:
        """
        모델 평가
        
        Returns:
            평가 지표 딕셔너리
        """
        print("\n" + "=" * 60)
        print("4. 모델 평가")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("학습된 모델이 없습니다.")
        
        # 정규화
        X_seq_norm, _ = self.feature_engineer.normalize_sequences(self.X_seq, fit=False)
        X_extra_norm, _ = self.feature_engineer.normalize_features(self.X_extra, fit=False)
        
        # 예측
        y_pred_norm = self.model.predict(X_seq_norm, X_extra_norm)
        y_pred = self.feature_engineer.denormalize_target(y_pred_norm)
        
        # 평가
        metrics = self.feature_engineer.create_evaluation_features(self.y, y_pred)
        
        print("\n평가 결과:")
        for name, value in metrics.items():
            print(f"  - {name}: {value:,.2f}")
        
        return {
            'metrics': metrics,
            'y_true': self.y,
            'y_pred': y_pred
        }
    
    def predict_next_period(self) -> Dict:
        """
        다음 기간 예측
        
        Returns:
            예측 결과 딕셔너리
        """
        freq_name = {'daily': '일', 'weekly': '주', 'monthly': '달'}[self.freq]
        
        print("\n" + "=" * 60)
        print(f"5. 다음 {freq_name} 예측")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("학습된 모델이 없습니다.")
        
        # 마지막 시퀀스 데이터
        last_seq = self.X_seq[-1:].copy()
        last_extra = self.X_extra[-1:].copy()
        
        # 정규화
        last_seq_norm, _ = self.feature_engineer.normalize_sequences(last_seq, fit=False)
        last_extra_norm, _ = self.feature_engineer.normalize_features(last_extra, fit=False)
        
        # 예측
        prediction_norm = self.model.predict(last_seq_norm, last_extra_norm)
        prediction = self.feature_engineer.denormalize_target(prediction_norm)
        
        # 마지막 기간 정보
        if 'period' in self.agg_data.columns:
            last_period = self.agg_data['period'].iloc[-1]
        elif 'date' in self.agg_data.columns:
            last_period = self.agg_data['date'].iloc[-1]
        else:
            last_period = self.agg_data['year_month'].iloc[-1]
            
        last_value = self.agg_data[self.target_col].iloc[-1]
        
        result = {
            'freq': self.freq,
            'last_period': str(last_period),
            'last_value': float(last_value),
            'predicted_value': float(prediction[0]),
            'change_percent': float((prediction[0] - last_value) / last_value * 100) if last_value != 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n예측 결과:")
        print(f"  - 마지막 기간 ({result['last_period']}): {result['last_value']:,.2f}")
        print(f"  - 다음 {freq_name} 예측: {result['predicted_value']:,.2f}")
        print(f"  - 변화율: {result['change_percent']:+.2f}%")
        
        return result
    
    def save_model(self) -> str:
        """
        모델 및 스케일러 저장
        
        Returns:
            저장 경로
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 모델 저장
        model_file = os.path.join(self.model_save_path, f'blinkit_model_{timestamp}.h5')
        self.model.save(model_file)
        
        # 스케일러 저장
        scaler_file = os.path.join(self.model_save_path, f'blinkit_scalers_{timestamp}.pkl')
        joblib.dump(self.feature_engineer, scaler_file)
        print(f"스케일러 저장 완료: {scaler_file}")
        
        return model_file
    
    def load_model(self, model_path: str = None) -> bool:
        """
        모델 로드
        
        Args:
            model_path: 모델 파일 경로 (None이면 최신 모델)
            
        Returns:
            로드 성공 여부
        """
        if model_path is None:
            # 최신 모델 찾기
            model_files = list(Path(self.model_save_path).glob("blinkit_model_*.h5"))
            if not model_files:
                print("저장된 Blinkit 모델이 없습니다.")
                return False
            model_path = str(max(model_files, key=os.path.getctime))
        
        # 모델 로드
        self.model = BlinkitDemandModel(self.config['model'])
        self.model.load(model_path)
        
        # 스케일러 로드
        scaler_path = model_path.replace('blinkit_model_', 'blinkit_scalers_').replace('.h5', '.pkl')
        if os.path.exists(scaler_path):
            self.feature_engineer = joblib.load(scaler_path)
            print(f"스케일러 로드 완료: {scaler_path}")
        
        return True
    
    def run_full_pipeline(self, 
                         retrain: bool = False,
                         verbose: int = 1) -> Dict:
        """
        전체 파이프라인 실행
        
        Args:
            retrain: 재학습 여부
            verbose: 출력 상세도
            
        Returns:
            실행 결과 딕셔너리
        """
        print("\n" + "=" * 60)
        print("       Blinkit 수요 예측 파이프라인 시작")
        print("=" * 60)
        start_time = datetime.now()
        
        try:
            # 1. 데이터 로딩 및 전처리
            self.load_and_prepare_data()
            
            # 2. 학습 데이터 준비
            self.prepare_training_data()
            
            # 데이터 충분성 확인
            min_months = self.config['automation']['min_data_months']
            if len(self.y) < min_months:
                return {
                    'status': 'insufficient_data',
                    'message': f'최소 {min_months}개월의 데이터가 필요합니다. (현재: {len(self.y)}개월)'
                }
            
            # 3. 모델 학습 또는 로드
            if retrain or not self.load_model():
                training_result = self.train_model(verbose=verbose)
                self.save_model()
            else:
                training_result = {'loaded_existing': True}
            
            # 4. 모델 평가
            eval_result = self.evaluate_model()
            
            # 5. 다음 기간 예측
            prediction_result = self.predict_next_period()
            
            # 결과 저장
            result = {
                'status': 'success',
                'training': training_result,
                'evaluation': eval_result['metrics'],
                'prediction': prediction_result,
                'execution_time': str(datetime.now() - start_time)
            }
            
            # 예측 결과 저장
            prediction_file = os.path.join(
                self.prediction_save_path,
                f"blinkit_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(prediction_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n예측 결과 저장: {prediction_file}")
            
            print("\n" + "=" * 60)
            print("       파이프라인 완료!")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            import traceback
            return {
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
    
    def visualize_results(self, 
                         eval_result: Dict = None,
                         save_path: Optional[str] = None):
        """
        결과 시각화
        
        Args:
            eval_result: 평가 결과 딕셔너리
            save_path: 저장 경로
        """
        import matplotlib.pyplot as plt
        
        if eval_result is None:
            eval_result = self.evaluate_model()
        
        y_true = eval_result['y_true']
        y_pred = eval_result['y_pred']
        
        # 월 정보 가져오기
        months = self.monthly_data['year_month'].iloc[self.sequence_length:].astype(str).values
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 실제 vs 예측
        axes[0, 0].plot(months, y_true, 'b-o', label='실제', markersize=4)
        axes[0, 0].plot(months, y_pred, 'r--s', label='예측', markersize=4)
        axes[0, 0].set_title('실제 vs 예측 매출', fontsize=12)
        axes[0, 0].set_xlabel('월')
        axes[0, 0].set_ylabel('매출')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 분석
        residuals = y_true - y_pred
        axes[0, 1].bar(range(len(residuals)), residuals, color='steelblue', alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('잔차 (실제 - 예측)', fontsize=12)
        axes[0, 1].set_xlabel('샘플')
        axes[0, 1].set_ylabel('잔차')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 산점도
        axes[1, 0].scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        axes[1, 0].set_title('예측 정확도', fontsize=12)
        axes[1, 0].set_xlabel('실제 값')
        axes[1, 0].set_ylabel('예측 값')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 오차 분포
        errors_percent = np.abs((y_true - y_pred) / y_true) * 100
        axes[1, 1].hist(errors_percent, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=np.median(errors_percent), color='r', linestyle='--', 
                          label=f'Median: {np.median(errors_percent):.1f}%')
        axes[1, 1].set_title('예측 오차 분포 (%)', fontsize=12)
        axes[1, 1].set_xlabel('오차 (%)')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"시각화 저장: {save_path}")
        
        plt.show()

