"""
단어 분석 및 매출 설명 실행 스크립트
- 단어와 매출 데이터로 단어 중요도 자동 학습
- 매출 변화에 대한 설명 생성
"""

import argparse
import sys
import os
import pandas as pd
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.word_analyzer import WordAnalyzer, SalesExplainer
from src.data_loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description='단어 분석 및 매출 설명')
    parser.add_argument(
        '--input-data',
        type=str,
        default='data/input_data/',
        help='단어 데이터 경로'
    )
    parser.add_argument(
        '--sales-data',
        type=str,
        default='data/sales/',
        help='매출 데이터 경로'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='gradient_boosting',
        choices=['ridge', 'lasso', 'random_forest', 'gradient_boosting'],
        help='학습 방법'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/word_scores.json',
        help='단어 점수 저장 경로'
    )
    parser.add_argument(
        '--explain',
        action='store_true',
        help='매출 설명 생성'
    )
    parser.add_argument(
        '--month',
        type=str,
        default=None,
        help='특정 월만 설명 (YYYY-MM)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("단어 분석 및 매출 설명 시스템")
    print("=" * 60)
    
    # 단어 분석기 초기화
    analyzer = WordAnalyzer()
    
    # 데이터 로더 초기화
    data_loader = DataLoader(args.input_data, args.sales_data)
    
    # 단어 데이터 로드
    print("\n1. 단어 데이터 로딩...")
    word_data = analyzer.load_word_data(args.input_data)
    
    if not word_data:
        print("오류: 단어 데이터를 찾을 수 없습니다.")
        print(f"경로: {args.input_data}")
        print("\n단어 데이터 형식:")
        print('  파일: YYYY-MM.json')
        print('  내용: ["단어1", "단어2", "단어3"] 또는 {"단어1": 0.5, "단어2": 0.3}')
        sys.exit(1)
    
    print(f"  로드된 월: {len(word_data)}개")
    
    # 매출 데이터 로드
    print("\n2. 매출 데이터 로딩...")
    sales_data = data_loader.load_sales_data()
    
    if sales_data.empty:
        print("오류: 매출 데이터를 찾을 수 없습니다.")
        print(f"경로: {args.sales_data}")
        sys.exit(1)
    
    print(f"  로드된 행: {len(sales_data)}개")
    
    # 학습 데이터 준비
    print("\n3. 학습 데이터 준비...")
    try:
        X, y, months = analyzer.prepare_training_data(word_data, sales_data)
    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)
    
    # 단어 점수 학습
    print(f"\n4. 단어 점수 학습 (방법: {args.method})...")
    word_scores = analyzer.learn_word_scores(X, y, method=args.method)
    
    # 단어 점수 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    analyzer.save_word_scores(args.output)
    
    # 상위/하위 단어 출력
    print("\n5. 분석 결과")
    print("\n📈 매출 상승 기여 단어 (Top 10):")
    for word, score in analyzer.get_top_words(10, 'positive'):
        print(f"  {word}: {score:.4f}")
    
    print("\n📉 매출 하락 기여 단어 (Top 10):")
    for word, score in analyzer.get_top_words(10, 'negative'):
        print(f"  {word}: {score:.4f}")
    
    # 매출 설명 생성
    if args.explain:
        print("\n6. 매출 설명 생성...")
        explainer = SalesExplainer(analyzer)
        
        if args.month:
            # 특정 월만 설명
            if args.month in word_data:
                words = word_data[args.month]
                sales_row = sales_data[sales_data['month_str'] == args.month] if 'month_str' in sales_data.columns else pd.DataFrame()
                actual_sales = sales_row['sales'].values[0] if not sales_row.empty else None
                
                if actual_sales:
                    report = explainer.explain_month(args.month, words, actual_sales, actual_sales)
                    print(report)
                else:
                    print(f"오류: {args.month}의 매출 데이터를 찾을 수 없습니다.")
            else:
                print(f"오류: {args.month}의 단어 데이터를 찾을 수 없습니다.")
        else:
            # 전체 월 설명
            report = explainer.explain_all_months(word_data, sales_data)
            print(report)
            
            # 보고서 저장
            report_path = args.output.replace('.json', '_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n보고서 저장: {report_path}")
    
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()

