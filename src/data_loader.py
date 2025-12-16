"""
데이터 로더 모듈
- 단어-점수 데이터 로딩
- 시계열 매출 데이터 로딩
- 데이터 전처리 및 결합
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class DataLoader:
    """데이터 로딩 및 전처리 클래스"""
    
    def __init__(self, input_data_path: str, sales_data_path: str):
        """
        Args:
            input_data_path: 단어-점수 데이터 경로
            sales_data_path: 매출 데이터 경로
        """
        self.input_data_path = Path(input_data_path)
        self.sales_data_path = Path(sales_data_path)
        
    def load_input_data(self, month: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        단어-점수 데이터 로딩
        
        Args:
            month: 특정 월만 로딩 (None이면 전체)
            
        Returns:
            {월: {단어: 점수}} 형태의 딕셔너리
        """
        input_data = {}
        
        if not self.input_data_path.exists():
            print(f"경고: {self.input_data_path} 경로가 존재하지 않습니다.")
            return input_data
        
        # JSON 파일들 로딩
        for file_path in self.input_data_path.glob("*.json"):
            file_month = file_path.stem  # 파일명에서 월 추출
            
            if month and file_month != month:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    input_data[file_month] = data
            except Exception as e:
                print(f"경고: {file_path} 로딩 실패: {e}")
        
        return input_data
    
    def load_sales_data(self) -> pd.DataFrame:
        """
        시계열 매출 데이터 로딩
        
        Returns:
            매출 데이터프레임 (컬럼: month, sales)
        """
        sales_data = None
        
        # CSV 파일 로딩
        csv_files = list(self.sales_data_path.glob("*.csv"))
        if csv_files:
            sales_data = pd.read_csv(csv_files[0])
            # month 컬럼이 있으면 datetime으로 변환
            if 'month' in sales_data.columns:
                sales_data['month'] = pd.to_datetime(sales_data['month'])
            elif 'date' in sales_data.columns:
                sales_data['month'] = pd.to_datetime(sales_data['date']).dt.to_period('M').dt.to_timestamp()
        else:
            print(f"경고: {self.sales_data_path}에 CSV 파일이 없습니다.")
            # 빈 데이터프레임 생성
            sales_data = pd.DataFrame(columns=['month', 'sales'])
        
        return sales_data
    
    def combine_data(self, input_data: Dict, sales_data: pd.DataFrame) -> pd.DataFrame:
        """
        단어-점수 데이터와 매출 데이터 결합
        
        Args:
            input_data: {월: {단어: 점수}} 형태
            sales_data: 매출 데이터프레임
            
        Returns:
            결합된 데이터프레임
        """
        # 매출 데이터 복사
        combined = sales_data.copy()
        
        # 단어-점수 데이터를 컬럼으로 변환
        all_words = set()
        for month_data in input_data.values():
            all_words.update(month_data.keys())
        
        # 각 단어에 대한 점수 컬럼 생성
        for word in all_words:
            combined[f'word_score_{word}'] = 0.0
        
        # 월별로 단어 점수 매핑
        for idx, row in combined.iterrows():
            month_str = self._get_month_string(row['month'])
            if month_str in input_data:
                for word, score in input_data[month_str].items():
                    combined.at[idx, f'word_score_{word}'] = score
        
        return combined
    
    def _get_month_string(self, month) -> str:
        """월을 문자열로 변환 (예: '2024-01')"""
        if isinstance(month, pd.Timestamp):
            return month.strftime('%Y-%m')
        elif isinstance(month, str):
            return month
        else:
            return str(month)
    
    def prepare_sequences(self, 
                         combined_data: pd.DataFrame, 
                         sequence_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 시퀀스 데이터 준비
        
        Args:
            combined_data: 결합된 데이터프레임
            sequence_length: 시퀀스 길이 (과거 몇 개월 사용할지)
            
        Returns:
            (X, y) - X: 입력 시퀀스, y: 타겟 매출
        """
        # 월별로 정렬
        combined_data = combined_data.sort_values('month').reset_index(drop=True)
        
        # 단어 점수 컬럼 추출
        word_columns = [col for col in combined_data.columns if col.startswith('word_score_')]
        feature_columns = word_columns + ['sales'] if 'sales' in combined_data.columns else word_columns
        
        X_sequences = []
        y_targets = []
        
        for i in range(sequence_length, len(combined_data)):
            # 과거 sequence_length 개월의 데이터
            sequence = combined_data.iloc[i-sequence_length:i][feature_columns].values
            # 다음 달 매출 (타겟)
            target = combined_data.iloc[i]['sales'] if 'sales' in combined_data.columns else 0
            
            X_sequences.append(sequence)
            y_targets.append(target)
        
        return np.array(X_sequences), np.array(y_targets)
    
    def get_latest_data_info(self) -> Dict:
        """최신 데이터 정보 반환"""
        input_data = self.load_input_data()
        sales_data = self.load_sales_data()
        
        return {
            'input_data_months': sorted(input_data.keys()) if input_data else [],
            'sales_data_months': sorted(sales_data['month'].dt.strftime('%Y-%m').unique().tolist()) if not sales_data.empty else [],
            'total_input_months': len(input_data),
            'total_sales_months': len(sales_data) if not sales_data.empty else 0
        }

