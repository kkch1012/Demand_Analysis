"""
단어 분석 및 자동 점수화 모듈
- 단어와 매출 데이터를 학습하여 단어별 중요도(점수) 자동 산출
- 매출 변화에 대한 설명 생성
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import json
from pathlib import Path


class WordAnalyzer:
    """단어 분석 및 자동 점수화 클래스"""
    
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_scores = {}
        self.feature_importance = {}
        self.model = None
        self.scaler = StandardScaler()
        
    def load_word_data(self, input_data_path: str) -> Dict[str, List[str]]:
        """
        월별 단어 데이터 로딩 (점수 없이 단어만)
        
        Args:
            input_data_path: 단어 데이터 경로
            
        Returns:
            {월: [단어1, 단어2, ...]} 형태의 딕셔너리
        """
        word_data = {}
        input_path = Path(input_data_path)
        
        if not input_path.exists():
            print(f"경고: {input_path} 경로가 존재하지 않습니다.")
            return word_data
        
        # JSON 파일들 로딩
        for file_path in input_path.glob("*.json"):
            file_month = file_path.stem
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 단어 리스트 또는 단어-점수 딕셔너리 처리
                    if isinstance(data, list):
                        # 단어 리스트인 경우
                        word_data[file_month] = data
                    elif isinstance(data, dict):
                        # 단어-점수 딕셔너리인 경우 (단어만 추출)
                        word_data[file_month] = list(data.keys())
                    else:
                        print(f"경고: {file_path} 형식이 올바르지 않습니다.")
            except Exception as e:
                print(f"경고: {file_path} 로딩 실패: {e}")
        
        return word_data
    
    def build_vocabulary(self, word_data: Dict[str, List[str]]) -> Dict[str, int]:
        """
        전체 단어 사전 구축
        
        Args:
            word_data: {월: [단어 리스트]}
            
        Returns:
            {단어: 인덱스} 딕셔너리
        """
        all_words = set()
        for words in word_data.values():
            all_words.update(words)
        
        self.word_to_index = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
        print(f"전체 단어 수: {len(self.word_to_index)}")
        return self.word_to_index
    
    def words_to_vector(self, words: List[str]) -> np.ndarray:
        """
        단어 리스트를 벡터로 변환 (원-핫 인코딩)
        
        Args:
            words: 단어 리스트
            
        Returns:
            단어 벡터 (0 또는 1)
        """
        vector = np.zeros(len(self.word_to_index))
        for word in words:
            if word in self.word_to_index:
                vector[self.word_to_index[word]] = 1.0
        return vector
    
    def prepare_training_data(self, 
                             word_data: Dict[str, List[str]], 
                             sales_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        학습 데이터 준비
        
        Args:
            word_data: {월: [단어 리스트]}
            sales_data: 매출 데이터프레임
            
        Returns:
            (X, y, months) - 단어 벡터, 매출 값, 월 리스트
        """
        # 사전 구축
        self.build_vocabulary(word_data)
        
        X_list = []
        y_list = []
        months_list = []
        
        # 매출 데이터에서 월 추출
        if 'month' in sales_data.columns:
            sales_data['month_str'] = pd.to_datetime(sales_data['month']).dt.strftime('%Y-%m')
        
        for month, words in word_data.items():
            # 해당 월의 매출 데이터 찾기
            if 'month_str' in sales_data.columns:
                sales_row = sales_data[sales_data['month_str'] == month]
            else:
                sales_row = pd.DataFrame()
            
            if not sales_row.empty and 'sales' in sales_row.columns:
                word_vector = self.words_to_vector(words)
                sales_value = sales_row['sales'].values[0]
                
                X_list.append(word_vector)
                y_list.append(sales_value)
                months_list.append(month)
        
        if not X_list:
            raise ValueError("매칭되는 데이터가 없습니다. 날짜 형식을 확인하세요.")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"학습 데이터: {len(X)}개월")
        return X, y, months_list
    
    def learn_word_scores(self, 
                         X: np.ndarray, 
                         y: np.ndarray,
                         method: str = 'gradient_boosting') -> Dict[str, float]:
        """
        단어별 점수(중요도) 학습
        
        Args:
            X: 단어 벡터
            y: 매출 값
            method: 학습 방법 ('ridge', 'lasso', 'random_forest', 'gradient_boosting')
            
        Returns:
            {단어: 점수} 딕셔너리
        """
        # 데이터 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # 모델 선택 및 학습
        if method == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif method == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif method == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif method == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"지원하지 않는 방법: {method}")
        
        self.model.fit(X_scaled, y)
        
        # 특성 중요도 추출
        if method in ['ridge', 'lasso']:
            importance = self.model.coef_
        else:
            importance = self.model.feature_importances_
        
        # 정규화된 점수 계산 (0~1 범위)
        importance_abs = np.abs(importance)
        if importance_abs.max() > 0:
            normalized_scores = importance_abs / importance_abs.max()
        else:
            normalized_scores = importance_abs
        
        # 단어별 점수 저장
        self.word_scores = {}
        self.feature_importance = {}
        
        for idx, score in enumerate(importance):
            word = self.index_to_word[idx]
            self.word_scores[word] = float(normalized_scores[idx])
            self.feature_importance[word] = {
                'score': float(normalized_scores[idx]),
                'raw_importance': float(importance[idx]),
                'direction': 'positive' if importance[idx] > 0 else 'negative'
            }
        
        # 점수 순으로 정렬
        self.word_scores = dict(sorted(
            self.word_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        print(f"\n=== 단어별 학습된 점수 ===")
        for i, (word, score) in enumerate(self.word_scores.items()):
            if i < 10:  # 상위 10개만 출력
                direction = self.feature_importance[word]['direction']
                print(f"  {word}: {score:.4f} ({direction})")
        
        return self.word_scores
    
    def explain_sales(self, 
                     month: str, 
                     words: List[str], 
                     predicted_sales: float,
                     actual_sales: Optional[float] = None) -> Dict:
        """
        매출에 대한 설명 생성
        
        Args:
            month: 월
            words: 해당 월의 단어 리스트
            predicted_sales: 예측 매출
            actual_sales: 실제 매출 (선택)
            
        Returns:
            설명 딕셔너리
        """
        # 단어별 영향도 계산
        word_impacts = []
        for word in words:
            if word in self.feature_importance:
                info = self.feature_importance[word]
                word_impacts.append({
                    'word': word,
                    'score': info['score'],
                    'direction': info['direction'],
                    'impact': 'high' if info['score'] > 0.7 else 'medium' if info['score'] > 0.3 else 'low'
                })
        
        # 영향도 순으로 정렬
        word_impacts.sort(key=lambda x: x['score'], reverse=True)
        
        # 긍정/부정 단어 분리
        positive_words = [w for w in word_impacts if w['direction'] == 'positive']
        negative_words = [w for w in word_impacts if w['direction'] == 'negative']
        
        # 설명 생성
        explanation = {
            'month': month,
            'predicted_sales': predicted_sales,
            'actual_sales': actual_sales,
            'total_words': len(words),
            'analyzed_words': len(word_impacts),
            'top_positive_factors': positive_words[:5],
            'top_negative_factors': negative_words[:5],
            'summary': self._generate_summary(positive_words, negative_words, predicted_sales)
        }
        
        return explanation
    
    def _generate_summary(self, 
                         positive_words: List[Dict], 
                         negative_words: List[Dict],
                         predicted_sales: float) -> str:
        """설명 요약 생성"""
        summary_parts = []
        
        if positive_words:
            top_positive = [w['word'] for w in positive_words[:3]]
            summary_parts.append(f"매출 상승 요인: {', '.join(top_positive)}")
        
        if negative_words:
            top_negative = [w['word'] for w in negative_words[:3]]
            summary_parts.append(f"매출 하락 요인: {', '.join(top_negative)}")
        
        if not summary_parts:
            summary_parts.append("분석된 주요 요인이 없습니다.")
        
        summary = " | ".join(summary_parts)
        return summary
    
    def save_word_scores(self, output_path: str):
        """학습된 단어 점수 저장"""
        output = {
            'word_scores': self.word_scores,
            'feature_importance': self.feature_importance
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"단어 점수 저장: {output_path}")
    
    def load_word_scores(self, input_path: str):
        """저장된 단어 점수 로드"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.word_scores = data['word_scores']
        self.feature_importance = data['feature_importance']
        
        print(f"단어 점수 로드: {input_path}")
    
    def get_top_words(self, n: int = 10, direction: str = 'all') -> List[Tuple[str, float]]:
        """
        상위 N개 단어 반환
        
        Args:
            n: 반환할 단어 수
            direction: 'all', 'positive', 'negative'
            
        Returns:
            [(단어, 점수)] 리스트
        """
        if direction == 'all':
            words = list(self.word_scores.items())[:n]
        elif direction == 'positive':
            words = [(w, s) for w, s in self.word_scores.items() 
                    if self.feature_importance[w]['direction'] == 'positive'][:n]
        elif direction == 'negative':
            words = [(w, s) for w, s in self.word_scores.items() 
                    if self.feature_importance[w]['direction'] == 'negative'][:n]
        else:
            words = list(self.word_scores.items())[:n]
        
        return words


class SalesExplainer:
    """매출 설명 생성 클래스"""
    
    def __init__(self, word_analyzer: WordAnalyzer):
        self.word_analyzer = word_analyzer
    
    def explain_month(self, 
                     month: str, 
                     words: List[str], 
                     predicted_sales: float,
                     actual_sales: Optional[float] = None) -> str:
        """
        특정 월의 매출 설명 생성
        
        Args:
            month: 월
            words: 단어 리스트
            predicted_sales: 예측 매출
            actual_sales: 실제 매출
            
        Returns:
            설명 문자열
        """
        explanation = self.word_analyzer.explain_sales(month, words, predicted_sales, actual_sales)
        
        report = []
        report.append(f"\n{'='*60}")
        report.append(f"📊 {month} 매출 분석 보고서")
        report.append(f"{'='*60}")
        
        if actual_sales:
            report.append(f"실제 매출: {actual_sales:,.0f}원")
        report.append(f"예측 매출: {predicted_sales:,.0f}원")
        
        if actual_sales:
            diff = predicted_sales - actual_sales
            diff_pct = (diff / actual_sales) * 100
            report.append(f"오차: {diff:,.0f}원 ({diff_pct:+.1f}%)")
        
        report.append(f"\n📈 매출 상승 요인 (Top 5):")
        for factor in explanation['top_positive_factors']:
            impact_emoji = '🔴' if factor['impact'] == 'high' else '🟡' if factor['impact'] == 'medium' else '🟢'
            report.append(f"  {impact_emoji} {factor['word']}: 중요도 {factor['score']:.2f}")
        
        if explanation['top_negative_factors']:
            report.append(f"\n📉 매출 하락 요인 (Top 5):")
            for factor in explanation['top_negative_factors']:
                impact_emoji = '🔴' if factor['impact'] == 'high' else '🟡' if factor['impact'] == 'medium' else '🟢'
                report.append(f"  {impact_emoji} {factor['word']}: 중요도 {factor['score']:.2f}")
        
        report.append(f"\n💡 요약: {explanation['summary']}")
        report.append(f"{'='*60}\n")
        
        return '\n'.join(report)
    
    def explain_all_months(self, 
                          word_data: Dict[str, List[str]], 
                          sales_data: pd.DataFrame,
                          predictions: Optional[Dict[str, float]] = None) -> str:
        """전체 월에 대한 분석 보고서 생성"""
        reports = []
        
        if 'month' in sales_data.columns:
            sales_data['month_str'] = pd.to_datetime(sales_data['month']).dt.strftime('%Y-%m')
        
        for month, words in sorted(word_data.items()):
            # 실제 매출 찾기
            if 'month_str' in sales_data.columns:
                sales_row = sales_data[sales_data['month_str'] == month]
                actual_sales = sales_row['sales'].values[0] if not sales_row.empty else None
            else:
                actual_sales = None
            
            # 예측 매출
            predicted_sales = predictions.get(month, actual_sales) if predictions else actual_sales
            
            if predicted_sales:
                report = self.explain_month(month, words, predicted_sales, actual_sales)
                reports.append(report)
        
        return '\n'.join(reports)

