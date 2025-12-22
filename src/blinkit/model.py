"""
Blinkit 수요 예측 모델 모듈
- 하이브리드 LSTM + Dense 모델
- 시계열 + 카테고리 특성 결합
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 경고 숨기기

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from typing import Dict, Tuple, Optional, List


class BlinkitDemandModel:
    """Blinkit 수요 예측 하이브리드 모델"""
    
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
                   n_sequence_features: int,
                   n_extra_features: int) -> keras.Model:
        """
        하이브리드 모델 구축
        
        Args:
            sequence_length: 시퀀스 길이
            n_sequence_features: 시퀀스 특성 개수
            n_extra_features: 추가 특성 개수
            
        Returns:
            컴파일된 모델
        """
        # 입력 1: 시계열 시퀀스 (LSTM)
        sequence_input = layers.Input(
            shape=(sequence_length, n_sequence_features),
            name='sequence_input'
        )
        
        # Bidirectional LSTM
        lstm_layers = sequence_input
        lstm_units = self.config['time_series']['lstm_units']
        dropout_rate = self.config['time_series']['dropout_rate']
        
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            
            # Bidirectional LSTM for better context
            lstm_layer = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=keras.regularizers.l2(0.01)
                ),
                name=f'bilstm_{i+1}'
            )(lstm_layers)
            
            lstm_layers = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(lstm_layer)
            lstm_layers = layers.Dropout(dropout_rate, name=f'dropout_lstm_{i+1}')(lstm_layers)
        
        # 입력 2: 추가 특성 (카테고리, 세그먼트 등)
        extra_input = layers.Input(
            shape=(n_extra_features,),
            name='extra_input'
        )
        
        extra_layers = extra_input
        category_units = self.config['category_features']['dense_units']
        category_dropout = self.config['category_features']['dropout_rate']
        
        for i, units in enumerate(category_units):
            extra_layers = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.01),
                name=f'extra_dense_{i+1}'
            )(extra_layers)
            extra_layers = layers.BatchNormalization(name=f'bn_extra_{i+1}')(extra_layers)
            extra_layers = layers.Dropout(category_dropout, name=f'dropout_extra_{i+1}')(extra_layers)
        
        # 특성 결합
        combined = layers.concatenate([lstm_layers, extra_layers], name='combined')
        
        # 결합된 레이어
        combined_layers = combined
        combined_units = self.config['combined']['dense_units']
        combined_dropout = self.config['combined']['dropout_rate']
        
        for i, units in enumerate(combined_units):
            combined_layers = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.01),
                name=f'combined_dense_{i+1}'
            )(combined_layers)
            combined_layers = layers.BatchNormalization(name=f'bn_combined_{i+1}')(combined_layers)
            combined_layers = layers.Dropout(combined_dropout, name=f'dropout_combined_{i+1}')(combined_layers)
        
        # 출력 레이어
        output = layers.Dense(
            self.config['combined']['output_units'],
            activation='linear',
            name='output'
        )(combined_layers)
        
        # 모델 생성
        model = models.Model(
            inputs=[sequence_input, extra_input],
            outputs=output,
            name='blinkit_demand_model'
        )
        
        # 컴파일
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config['training']['learning_rate']
            ),
            loss='huber',  # Huber loss for robustness
            metrics=['mae', 'mape']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, 
                     validation_data: bool = True,
                     checkpoint_path: Optional[str] = None) -> List[callbacks.Callback]:
        """
        학습 콜백 생성
        
        Args:
            validation_data: 검증 데이터 사용 여부
            checkpoint_path: 체크포인트 저장 경로
            
        Returns:
            콜백 리스트
        """
        monitor = 'val_loss' if validation_data else 'loss'
        
        callback_list = [
            # Early Stopping
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            # Learning Rate Reduction
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
        ]
        
        # Model Checkpoint
        if checkpoint_path:
            callback_list.append(
                callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor=monitor,
                    save_best_only=True,
                    verbose=1
                )
            )
        
        return callback_list
    
    def train(self, 
              X_sequence: np.ndarray,
              X_extra: np.ndarray,
              y: np.ndarray,
              validation_split: float = 0.2,
              verbose: int = 1) -> keras.callbacks.History:
        """
        모델 학습
        
        Args:
            X_sequence: 시퀀스 입력 데이터
            X_extra: 추가 특성 입력 데이터
            y: 타겟
            validation_split: 검증 데이터 비율
            verbose: 출력 상세도
            
        Returns:
            학습 히스토리
        """
        # 데이터 분할
        split_idx = int(len(X_sequence) * (1 - validation_split))
        
        X_seq_train, X_seq_val = X_sequence[:split_idx], X_sequence[split_idx:]
        X_extra_train, X_extra_val = X_extra[:split_idx], X_extra[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        validation_data = ([X_seq_val, X_extra_val], y_val)
        
        # 콜백 설정
        callback_list = self.get_callbacks(validation_data=True)
        
        print(f"\n학습 데이터: {len(X_seq_train)}개, 검증 데이터: {len(X_seq_val)}개")
        
        self.history = self.model.fit(
            [X_seq_train, X_extra_train],
            y_train,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, 
                X_sequence: np.ndarray,
                X_extra: np.ndarray) -> np.ndarray:
        """
        예측 수행
        
        Args:
            X_sequence: 시퀀스 입력 데이터
            X_extra: 추가 특성 입력 데이터
            
        Returns:
            예측 값
        """
        predictions = self.model.predict(
            [X_sequence, X_extra], 
            verbose=0
        )
        return predictions.flatten()
    
    def summary(self):
        """모델 요약 출력"""
        if self.model:
            self.model.summary()
    
    def save(self, filepath: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"모델 저장 완료: {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        self.model = keras.models.load_model(filepath)
        print(f"모델 로드 완료: {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        학습 히스토리 시각화
        
        Args:
            save_path: 저장 경로 (선택)
        """
        if self.history is None:
            print("학습 히스토리가 없습니다.")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"학습 히스토리 저장: {save_path}")
        
        plt.show()

