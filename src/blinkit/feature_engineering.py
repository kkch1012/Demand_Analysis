"""
Blinkit 특성 엔지니어링 모듈
- 시계열 특성 추출
- 특성 정규화
- 지연 특성 및 이동평균
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BlinkitFeatureEngineer:
    """Blinkit 특성 엔지니어링 클래스"""
    
    def __init__(self):
        self.sequence_scaler = StandardScaler()
        self.extra_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.is_fitted = False
        
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        시간 관련 특성 추가
        
        Args:
            data: 데이터프레임 (month 컬럼 포함)
            
        Returns:
            시간 특성이 추가된 데이터프레임
        """
        df = data.copy()
        
        if 'month' in df.columns:
            df['month_num'] = df['month'].dt.month
            df['quarter'] = df['month'].dt.quarter
            df['year'] = df['month'].dt.year
            
            # 주기적 인코딩
            df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
            df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
            df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        return df
    
    def add_lag_features(self, 
                        data: pd.DataFrame, 
                        target_col: str,
                        lags: List[int] = [1, 2, 3, 6]) -> pd.DataFrame:
        """
        지연 특성 추가
        
        Args:
            data: 데이터프레임
            target_col: 타겟 컬럼
            lags: 지연 기간 리스트
            
        Returns:
            지연 특성이 추가된 데이터프레임
        """
        df = data.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self,
                            data: pd.DataFrame,
                            target_col: str,
                            windows: List[int] = [2, 3, 6]) -> pd.DataFrame:
        """
        이동 통계 특성 추가
        
        Args:
            data: 데이터프레임
            target_col: 타겟 컬럼
            windows: 윈도우 크기 리스트
            
        Returns:
            이동 통계 특성이 추가된 데이터프레임
        """
        df = data.copy()
        
        for window in windows:
            df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_max_{window}'] = df[target_col].rolling(window=window).max()
        
        return df
    
    def add_growth_features(self,
                           data: pd.DataFrame,
                           target_col: str) -> pd.DataFrame:
        """
        성장률 특성 추가
        
        Args:
            data: 데이터프레임
            target_col: 타겟 컬럼
            
        Returns:
            성장률 특성이 추가된 데이터프레임
        """
        df = data.copy()
        
        # 전월 대비 성장률
        df[f'{target_col}_mom_growth'] = df[target_col].pct_change()
        
        # 전년 동월 대비 성장률 (데이터가 충분한 경우)
        if len(df) > 12:
            df[f'{target_col}_yoy_growth'] = df[target_col].pct_change(periods=12)
        
        # 무한대 및 결측치 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def normalize_sequences(self, 
                           X_train: np.ndarray, 
                           X_test: Optional[np.ndarray] = None,
                           fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        시퀀스 데이터 정규화
        
        Args:
            X_train: 학습 시퀀스 데이터 (3D: samples, timesteps, features)
            X_test: 테스트 시퀀스 데이터 (선택)
            fit: 스케일러 학습 여부
            
        Returns:
            정규화된 데이터
        """
        n_samples, n_timesteps, n_features = X_train.shape
        
        # 3D -> 2D
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        if fit:
            X_train_normalized = self.sequence_scaler.fit_transform(X_train_reshaped)
        else:
            X_train_normalized = self.sequence_scaler.transform(X_train_reshaped)
        
        # 2D -> 3D
        X_train_normalized = X_train_normalized.reshape(n_samples, n_timesteps, n_features)
        
        if X_test is not None:
            n_test_samples = X_test.shape[0]
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_normalized = self.sequence_scaler.transform(X_test_reshaped)
            X_test_normalized = X_test_normalized.reshape(n_test_samples, n_timesteps, n_features)
            return X_train_normalized, X_test_normalized
        
        return X_train_normalized, None
    
    def normalize_features(self, 
                          X_train: np.ndarray, 
                          X_test: Optional[np.ndarray] = None,
                          fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        추가 특성 정규화
        
        Args:
            X_train: 학습 데이터
            X_test: 테스트 데이터 (선택)
            fit: 스케일러 학습 여부
            
        Returns:
            정규화된 데이터
        """
        if fit:
            X_train_normalized = self.extra_scaler.fit_transform(X_train)
        else:
            X_train_normalized = self.extra_scaler.transform(X_train)
        
        if X_test is not None:
            X_test_normalized = self.extra_scaler.transform(X_test)
            return X_train_normalized, X_test_normalized
        
        return X_train_normalized, None
    
    def normalize_target(self, 
                        y_train: np.ndarray, 
                        y_test: Optional[np.ndarray] = None,
                        fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        타겟 값 정규화
        
        Args:
            y_train: 학습 타겟
            y_test: 테스트 타겟 (선택)
            fit: 스케일러 학습 여부
            
        Returns:
            정규화된 타겟
        """
        y_train_reshaped = y_train.reshape(-1, 1)
        
        if fit:
            y_train_normalized = self.target_scaler.fit_transform(y_train_reshaped).flatten()
            self.is_fitted = True
        else:
            y_train_normalized = self.target_scaler.transform(y_train_reshaped).flatten()
        
        if y_test is not None:
            y_test_reshaped = y_test.reshape(-1, 1)
            y_test_normalized = self.target_scaler.transform(y_test_reshaped).flatten()
            return y_train_normalized, y_test_normalized
        
        return y_train_normalized, None
    
    def denormalize_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        정규화된 타겟 값을 원래 스케일로 변환
        
        Args:
            y_normalized: 정규화된 타겟
            
        Returns:
            원래 스케일의 타겟
        """
        if not self.is_fitted:
            raise ValueError("Scaler가 학습되지 않았습니다.")
        
        y_reshaped = y_normalized.reshape(-1, 1)
        y_original = self.target_scaler.inverse_transform(y_reshaped).flatten()
        
        return y_original
    
    def create_evaluation_features(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> dict:
        """
        예측 평가 지표 계산
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
            
        Returns:
            평가 지표 딕셔너리
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

