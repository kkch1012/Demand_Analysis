"""
Blinkit 데이터 로더 모듈
- Blinkit 주문 데이터 로딩
- 월별 집계 및 전처리
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


class BlinkitDataLoader:
    """Blinkit 데이터 로딩 및 전처리 클래스"""
    
    def __init__(self, raw_data_path: str):
        """
        Args:
            raw_data_path: 원본 CSV 데이터 경로
        """
        self.raw_data_path = Path(raw_data_path)
        self.df = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        원본 데이터 로딩
        
        Returns:
            원본 데이터프레임
        """
        print(f"데이터 로딩 중: {self.raw_data_path}")
        
        self.df = pd.read_csv(self.raw_data_path)
        
        # 날짜 컬럼 변환
        self.df['order_date'] = pd.to_datetime(self.df['order_date'])
        self.df['year_month'] = self.df['order_date'].dt.to_period('M')
        
        print(f"총 {len(self.df):,}개 주문 로드 완료")
        print(f"기간: {self.df['order_date'].min()} ~ {self.df['order_date'].max()}")
        
        return self.df
    
    def aggregate_data(self, freq: str = 'monthly') -> pd.DataFrame:
        """
        데이터 집계 (일간/주간/월간)
        
        Args:
            freq: 'daily', 'weekly', 'monthly'
            
        Returns:
            집계된 데이터프레임
        """
        if self.df is None:
            self.load_raw_data()
        
        # 집계 기준 컬럼 생성
        if freq == 'daily':
            self.df['period'] = self.df['order_date'].dt.date
            period_name = '일'
        elif freq == 'weekly':
            self.df['period'] = self.df['order_date'].dt.to_period('W').dt.start_time
            period_name = '주'
        else:  # monthly
            self.df['period'] = self.df['year_month']
            period_name = '월'
        
        # 집계
        agg_data = self.df.groupby('period').agg({
            'order_id': 'count',                    # 주문 수
            'order_total': ['sum', 'mean', 'std'],  # 매출 통계
            'quantity': ['sum', 'mean'],            # 수량 통계
            'customer_id_x': 'nunique',             # 고객 수
            'delivery_time_minutes': ['mean', 'std'],  # 배송 시간
            'rating': 'mean',                       # 평균 평점
        }).reset_index()
        
        # 컬럼명 정리
        agg_data.columns = [
            'period',
            'order_count',
            'total_sales', 'avg_order_value', 'std_order_value',
            'total_quantity', 'avg_quantity',
            'unique_customers',
            'avg_delivery_time', 'std_delivery_time',
            'avg_rating'
        ]
        
        # 결측치 처리
        agg_data = agg_data.fillna(0)
        
        # 정렬
        agg_data = agg_data.sort_values('period').reset_index(drop=True)
        
        # period를 datetime으로 변환
        if freq == 'daily':
            agg_data['date'] = pd.to_datetime(agg_data['period'])
        elif freq == 'weekly':
            agg_data['date'] = pd.to_datetime(agg_data['period'])
        else:
            agg_data['date'] = agg_data['period'].dt.to_timestamp()
        
        # 시간 특성 추가
        agg_data['day_of_week'] = agg_data['date'].dt.dayofweek
        agg_data['month'] = agg_data['date'].dt.month
        agg_data['week_of_year'] = agg_data['date'].dt.isocalendar().week.astype(int)
        
        print(f"{period_name}별 집계 완료: {len(agg_data)}개 기간")
        
        return agg_data
    
    def aggregate_monthly(self) -> pd.DataFrame:
        """월별 집계 (하위 호환성)"""
        data = self.aggregate_data('monthly')
        data['year_month'] = data['period']
        data['month'] = data['date']
        return data
    
    def get_category_features(self, freq: str = 'monthly') -> pd.DataFrame:
        """
        카테고리별 통계
        
        Args:
            freq: 'daily', 'weekly', 'monthly'
            
        Returns:
            카테고리별 매출 피벗 테이블
        """
        if self.df is None:
            self.load_raw_data()
        
        # 집계 기준 설정
        if freq == 'daily':
            self.df['_period'] = self.df['order_date'].dt.date
        elif freq == 'weekly':
            self.df['_period'] = self.df['order_date'].dt.to_period('W').dt.start_time
        else:
            self.df['_period'] = self.df['year_month']
        
        # 카테고리별 집계
        category_agg = self.df.groupby(['_period', 'category']).agg({
            'order_total': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        # 피벗 테이블 생성
        category_pivot = category_agg.pivot_table(
            index='_period',
            columns='category',
            values='order_total',
            fill_value=0
        )
        
        # 컬럼명에 접두사 추가
        category_pivot.columns = [f'cat_sales_{col}' for col in category_pivot.columns]
        
        return category_pivot.reset_index().rename(columns={'_period': 'period'})
    
    def get_segment_features(self, freq: str = 'monthly') -> pd.DataFrame:
        """
        고객 세그먼트별 통계
        
        Args:
            freq: 'daily', 'weekly', 'monthly'
        """
        if self.df is None:
            self.load_raw_data()
        
        if freq == 'daily':
            self.df['_period'] = self.df['order_date'].dt.date
        elif freq == 'weekly':
            self.df['_period'] = self.df['order_date'].dt.to_period('W').dt.start_time
        else:
            self.df['_period'] = self.df['year_month']
        
        segment_agg = self.df.groupby(['_period', 'customer_segment']).agg({
            'order_total': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        segment_pivot = segment_agg.pivot_table(
            index='_period',
            columns='customer_segment',
            values='order_total',
            fill_value=0
        )
        
        segment_pivot.columns = [f'seg_sales_{col}' for col in segment_pivot.columns]
        
        return segment_pivot.reset_index().rename(columns={'_period': 'period'})
    
    def get_delivery_features(self, freq: str = 'monthly') -> pd.DataFrame:
        """배송 상태별 통계"""
        if self.df is None:
            self.load_raw_data()
        
        if freq == 'daily':
            self.df['_period'] = self.df['order_date'].dt.date
        elif freq == 'weekly':
            self.df['_period'] = self.df['order_date'].dt.to_period('W').dt.start_time
        else:
            self.df['_period'] = self.df['year_month']
        
        delivery_agg = self.df.groupby(['_period', 'delivery_status_x']).agg({
            'order_id': 'count'
        }).reset_index()
        
        delivery_pivot = delivery_agg.pivot_table(
            index='_period',
            columns='delivery_status_x',
            values='order_id',
            fill_value=0
        )
        
        delivery_pivot.columns = [f'delivery_{col.replace(" ", "_")}' for col in delivery_pivot.columns]
        
        return delivery_pivot.reset_index().rename(columns={'_period': 'period'})
    
    def get_sentiment_features(self, freq: str = 'monthly') -> pd.DataFrame:
        """감성 분석별 통계"""
        if self.df is None:
            self.load_raw_data()
        
        if freq == 'daily':
            self.df['_period'] = self.df['order_date'].dt.date
        elif freq == 'weekly':
            self.df['_period'] = self.df['order_date'].dt.to_period('W').dt.start_time
        else:
            self.df['_period'] = self.df['year_month']
        
        sentiment_agg = self.df.groupby(['_period', 'sentiment']).agg({
            'order_id': 'count'
        }).reset_index()
        
        sentiment_pivot = sentiment_agg.pivot_table(
            index='_period',
            columns='sentiment',
            values='order_id',
            fill_value=0
        )
        
        sentiment_pivot.columns = [f'sentiment_{col}' for col in sentiment_pivot.columns]
        
        return sentiment_pivot.reset_index().rename(columns={'_period': 'period'})
    
    def get_payment_features(self, freq: str = 'monthly') -> pd.DataFrame:
        """결제 방식별 통계"""
        if self.df is None:
            self.load_raw_data()
        
        if freq == 'daily':
            self.df['_period'] = self.df['order_date'].dt.date
        elif freq == 'weekly':
            self.df['_period'] = self.df['order_date'].dt.to_period('W').dt.start_time
        else:
            self.df['_period'] = self.df['year_month']
        
        payment_agg = self.df.groupby(['_period', 'payment_method']).agg({
            'order_total': 'sum'
        }).reset_index()
        
        payment_pivot = payment_agg.pivot_table(
            index='_period',
            columns='payment_method',
            values='order_total',
            fill_value=0
        )
        
        payment_pivot.columns = [f'payment_{col}' for col in payment_pivot.columns]
        
        return payment_pivot.reset_index().rename(columns={'_period': 'period'})
    
    def prepare_full_dataset(self, freq: str = 'monthly') -> pd.DataFrame:
        """
        전체 데이터셋 준비 (집계 + 모든 피처)
        
        Args:
            freq: 'daily', 'weekly', 'monthly'
            
        Returns:
            전체 피처가 포함된 데이터프레임
        """
        # 기본 집계
        agg_data = self.aggregate_data(freq)
        
        # 카테고리 피처
        category_features = self.get_category_features(freq)
        agg_data = agg_data.merge(category_features, on='period', how='left')
        
        # 세그먼트 피처
        segment_features = self.get_segment_features(freq)
        agg_data = agg_data.merge(segment_features, on='period', how='left')
        
        # 배송 피처
        delivery_features = self.get_delivery_features(freq)
        agg_data = agg_data.merge(delivery_features, on='period', how='left')
        
        # 감성 피처
        sentiment_features = self.get_sentiment_features(freq)
        agg_data = agg_data.merge(sentiment_features, on='period', how='left')
        
        # 결제 피처
        payment_features = self.get_payment_features(freq)
        agg_data = agg_data.merge(payment_features, on='period', how='left')
        
        # 결측치 처리
        agg_data = agg_data.fillna(0)
        
        # 하위 호환성을 위해 year_month 컬럼 추가
        if freq == 'monthly':
            agg_data['year_month'] = agg_data['period']
        
        print(f"전체 데이터셋 준비 완료: {agg_data.shape}")
        print(f"피처 수: {len(agg_data.columns)}")
        
        return agg_data
    
    def prepare_sequences(self, 
                         data: pd.DataFrame, 
                         target_col: str,
                         sequence_length: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        시계열 시퀀스 데이터 준비
        
        Args:
            data: 전체 데이터프레임
            target_col: 타겟 컬럼명
            sequence_length: 시퀀스 길이
            
        Returns:
            (X_sequences, X_features, y) - 시퀀스 입력, 추가 피처, 타겟
        """
        # 정렬 (period 또는 date 기준)
        if 'period' in data.columns:
            data = data.sort_values('period').reset_index(drop=True)
        elif 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)
        elif 'year_month' in data.columns:
            data = data.sort_values('year_month').reset_index(drop=True)
        
        # 피처 컬럼 선택 (기본 피처)
        sequence_features = [
            'order_count', 'total_sales', 'avg_order_value', 
            'total_quantity', 'unique_customers',
            'avg_delivery_time', 'avg_rating'
        ]
        # 존재하는 피처만 선택
        sequence_features = [f for f in sequence_features if f in data.columns]
        
        # 추가 피처 컬럼 (카테고리, 세그먼트 등)
        extra_features = [col for col in data.columns 
                        if col.startswith(('cat_', 'seg_', 'delivery_', 'sentiment_', 'payment_'))]
        
        X_sequences = []
        X_extra = []
        y_targets = []
        
        for i in range(sequence_length, len(data)):
            # 과거 sequence_length 기간의 시계열 데이터
            seq_data = data.iloc[i-sequence_length:i][sequence_features].values
            X_sequences.append(seq_data)
            
            # 현재 시점의 추가 피처
            extra_data = data.iloc[i][extra_features].values if extra_features else np.array([0])
            X_extra.append(extra_data)
            
            # 타겟
            target = data.iloc[i][target_col]
            y_targets.append(target)
        
        X_sequences = np.array(X_sequences)
        X_extra = np.array(X_extra)
        y_targets = np.array(y_targets)
        
        print(f"시퀀스 데이터 준비 완료:")
        print(f"  - X_sequences: {X_sequences.shape}")
        print(f"  - X_extra: {X_extra.shape}")
        print(f"  - y: {y_targets.shape}")
        
        return X_sequences, X_extra, y_targets
    
    def get_data_info(self) -> Dict:
        """데이터 정보 반환"""
        if self.df is None:
            self.load_raw_data()
        
        return {
            'total_orders': len(self.df),
            'date_range': {
                'start': str(self.df['order_date'].min()),
                'end': str(self.df['order_date'].max())
            },
            'categories': self.df['category'].unique().tolist(),
            'segments': self.df['customer_segment'].unique().tolist(),
            'total_months': self.df['year_month'].nunique()
        }

