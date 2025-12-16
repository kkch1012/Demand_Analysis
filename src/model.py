"""
모델 정의 모듈
- 하이브리드 모델 (시계열 LSTM + 텍스트 특성)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Dict, Tuple, Optional
import os


class SalesPredictionModel:
    """매출 예측 하이브리드 모델"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 모델 설정 딕셔너리
        """
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, 
                   sequence_length: int,
                   n_time_features: int,
                   n_word_features: int) -> keras.Model:
        """
        하이브리드 모델 구축
        
        Args:
            sequence_length: 시퀀스 길이
            n_time_features: 시계열 특성 개수
            n_word_features: 단어 특성 개수
            
        Returns:
            컴파일된 모델
        """
        # 입력 1: 시계열 시퀀스 (LSTM)
        time_series_input = layers.Input(
            shape=(sequence_length, n_time_features),
            name='time_series_input'
        )
        
        lstm_layers = time_series_input
        for i, units in enumerate(self.config['time_series']['lstm_units']):
            lstm_layers = layers.LSTM(
                units,
                return_sequences=(i < len(self.config['time_series']['lstm_units']) - 1),
                name=f'lstm_{i+1}'
            )(lstm_layers)
            lstm_layers = layers.Dropout(
                self.config['time_series']['dropout_rate'],
                name=f'dropout_lstm_{i+1}'
            )(lstm_layers)
        
        # 입력 2: 단어-점수 특성 (Dense)
        word_input = layers.Input(
            shape=(n_word_features,),
            name='word_input'
        )
        
        word_layers = word_input
        for i, units in enumerate(self.config['text_features']['dense_units']):
            word_layers = layers.Dense(
                units,
                activation='relu',
                name=f'word_dense_{i+1}'
            )(word_layers)
            word_layers = layers.Dropout(
                self.config['text_features'].get('dropout_rate', 0.2),
                name=f'dropout_word_{i+1}'
            )(word_layers)
        
        # 특성 결합
        if len(self.config['time_series']['lstm_units']) > 0:
            combined = layers.concatenate([lstm_layers, word_layers], name='combined')
        else:
            combined = word_layers
        
        # 결합된 레이어
        dense_layers = combined
        for i, units in enumerate(self.config['combined']['dense_units']):
            dense_layers = layers.Dense(
                units,
                activation='relu',
                name=f'combined_dense_{i+1}'
            )(dense_layers)
            dense_layers = layers.Dropout(
                self.config['combined'].get('dropout_rate', 0.2),
                name=f'dropout_combined_{i+1}'
            )(dense_layers)
        
        # 출력 레이어
        output = layers.Dense(
            self.config['combined']['output_units'],
            activation='linear',
            name='output'
        )(dense_layers)
        
        # 모델 생성
        if len(self.config['time_series']['lstm_units']) > 0:
            model = models.Model(
                inputs=[time_series_input, word_input],
                outputs=output,
                name='sales_prediction_model'
            )
        else:
            model = models.Model(
                inputs=word_input,
                outputs=output,
                name='sales_prediction_model'
            )
        
        # 컴파일
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config['training']['learning_rate']
            ),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        return model
    
    def train(self, 
              X_time: np.ndarray,
              X_word: np.ndarray,
              y: np.ndarray,
              validation_data: Optional[Tuple] = None,
              verbose: int = 1) -> keras.callbacks.History:
        """
        모델 학습
        
        Args:
            X_time: 시계열 입력 데이터
            X_word: 단어 특성 입력 데이터
            y: 타겟 매출
            validation_data: 검증 데이터 (선택)
            verbose: 출력 상세도
            
        Returns:
            학습 히스토리
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if len(self.config['time_series']['lstm_units']) > 0:
            inputs = [X_time, X_word]
        else:
            inputs = X_word
        
        self.history = self.model.fit(
            inputs,
            y,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, 
                X_time: np.ndarray,
                X_word: np.ndarray) -> np.ndarray:
        """
        예측 수행
        
        Args:
            X_time: 시계열 입력 데이터
            X_word: 단어 특성 입력 데이터
            
        Returns:
            예측된 매출 값
        """
        if len(self.config['time_series']['lstm_units']) > 0:
            inputs = [X_time, X_word]
        else:
            inputs = X_word
        
        predictions = self.model.predict(inputs, verbose=0)
        return predictions.flatten()
    
    def save(self, filepath: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"모델이 저장되었습니다: {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        self.model = keras.models.load_model(filepath)
        print(f"모델이 로드되었습니다: {filepath}")

