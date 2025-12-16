"""
특성 엔지니어링 모듈
- 단어-점수 특성 처리
- 시계열 특성 추출
- 특성 정규화
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureEngineer:
    """특성 엔지니어링 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.sales_scaler = MinMaxScaler()
        self.word_columns = []
        
    def extract_word_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        단어-점수 특성 추출
        
        Args:
            data: 데이터프레임
            
        Returns:
            (단어 특성 배열, 단어 컬럼명 리스트)
        """
        word_columns = [col for col in data.columns if col.startswith('word_score_')]
        self.word_columns = word_columns
        
        if not word_columns:
            return np.array([]), []
        
        word_features = data[word_columns].values
        return word_features, word_columns
    
    def extract_time_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        시계열 특성 추출
        
        Args:
            data: 데이터프레임 (month 컬럼 포함)
            
        Returns:
            시계열 특성 배열 (월, 분기, 연도 등)
        """
        if 'month' not in data.columns:
            return np.array([])
        
        time_features = []
        
        for month in data['month']:
            if isinstance(month, pd.Timestamp):
                features = [
                    month.month,  # 월 (1-12)
                    month.quarter,  # 분기 (1-4)
                    month.year,  # 연도
                    np.sin(2 * np.pi * month.month / 12),  # 월의 주기적 특성
                    np.cos(2 * np.pi * month.month / 12),
                ]
            else:
                features = [0, 0, 0, 0, 0]
            
            time_features.append(features)
        
        return np.array(time_features)
    
    def extract_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 3, 6, 12]) -> np.ndarray:
        """
        지연 특성 추출 (과거 매출 값들)
        
        Args:
            data: 데이터프레임
            lags: 지연 개월 수 리스트
            
        Returns:
            지연 특성 배열
        """
        if 'sales' not in data.columns:
            return np.array([])
        
        lag_features = []
        
        for lag in lags:
            lag_values = data['sales'].shift(lag).fillna(0).values
            lag_features.append(lag_values)
        
        return np.array(lag_features).T if lag_features else np.array([])
    
    def extract_statistical_features(self, 
                                   sequences: np.ndarray, 
                                   window_size: int = 3) -> np.ndarray:
        """
        통계적 특성 추출 (이동평균, 표준편차 등)
        
        Args:
            sequences: 시퀀스 데이터
            window_size: 윈도우 크기
            
        Returns:
            통계 특성 배열
        """
        if len(sequences) == 0:
            return np.array([])
        
        stats_features = []
        
        for i in range(len(sequences)):
            if i < window_size:
                window_data = sequences[:i+1]
            else:
                window_data = sequences[i-window_size+1:i+1]
            
            # 매출 컬럼이 마지막에 있다고 가정
            if window_data.shape[1] > 0:
                sales_col = window_data[:, -1] if window_data.shape[1] > 0 else window_data.flatten()
                
                features = [
                    np.mean(sales_col),  # 평균
                    np.std(sales_col) if len(sales_col) > 1 else 0,  # 표준편차
                    np.min(sales_col),  # 최소값
                    np.max(sales_col),  # 최대값
                ]
            else:
                features = [0, 0, 0, 0]
            
            stats_features.append(features)
        
        return np.array(stats_features)
    
    def normalize_features(self, 
                          X_train: np.ndarray, 
                          X_test: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        특성 정규화
        
        Args:
            X_train: 학습 데이터
            X_test: 테스트 데이터 (선택)
            
        Returns:
            정규화된 데이터
        """
        # 3D 배열인 경우 (시퀀스 데이터)
        if len(X_train.shape) == 3:
            # 각 시퀀스의 각 타임스텝을 정규화
            n_samples, n_timesteps, n_features = X_train.shape
            X_train_reshaped = X_train.reshape(-1, n_features)
            X_train_normalized = self.scaler.fit_transform(X_train_reshaped)
            X_train_normalized = X_train_normalized.reshape(n_samples, n_timesteps, n_features)
            
            if X_test is not None:
                n_test_samples = X_test.shape[0]
                X_test_reshaped = X_test.reshape(-1, n_features)
                X_test_normalized = self.scaler.transform(X_test_reshaped)
                X_test_normalized = X_test_normalized.reshape(n_test_samples, n_timesteps, n_features)
                return X_train_normalized, X_test_normalized
            
            return X_train_normalized, None
        else:
            # 2D 배열인 경우
            X_train_normalized = self.scaler.fit_transform(X_train)
            
            if X_test is not None:
                X_test_normalized = self.scaler.transform(X_test)
                return X_train_normalized, X_test_normalized
            
            return X_train_normalized, None
    
    def normalize_sales(self, y_train: np.ndarray, y_test: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        매출 값 정규화
        
        Args:
            y_train: 학습 타겟
            y_test: 테스트 타겟 (선택)
            
        Returns:
            정규화된 타겟
        """
        y_train_reshaped = y_train.reshape(-1, 1)
        y_train_normalized = self.sales_scaler.fit_transform(y_train_reshaped).flatten()
        
        if y_test is not None:
            y_test_reshaped = y_test.reshape(-1, 1)
            y_test_normalized = self.sales_scaler.transform(y_test_reshaped).flatten()
            return y_train_normalized, y_test_normalized
        
        return y_train_normalized, None
    
    def denormalize_sales(self, y_normalized: np.ndarray) -> np.ndarray:
        """정규화된 매출 값을 원래 스케일로 변환"""
        y_reshaped = y_normalized.reshape(-1, 1)
        y_original = self.sales_scaler.inverse_transform(y_reshaped).flatten()
        return y_original

