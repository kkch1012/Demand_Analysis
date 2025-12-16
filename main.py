"""
메인 실행 스크립트
자동화 파이프라인 실행
"""

import argparse
import sys
import numpy as np
from src.pipeline import AutomatedPipeline


def main():
    parser = argparse.ArgumentParser(description='매출 예측 자동화 파이프라인')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='강제 재학습'
    )
    parser.add_argument(
        '--predict-only',
        action='store_true',
        help='예측만 수행 (학습 생략)'
    )
    parser.add_argument(
        '--next-month',
        type=str,
        default=None,
        help='예측할 월 지정 (YYYY-MM 형식, 기본값: 다음 달)'
    )
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = AutomatedPipeline(config_path=args.config)
    
    if args.predict_only:
        # 예측만 수행
        if not pipeline.load_latest_model():
            print("오류: 학습된 모델이 없습니다. 먼저 학습을 실행하세요.")
            sys.exit(1)
        
        # 데이터 준비
        X_time, X_word, y, data_info = pipeline.prepare_data()
        
        # 예측
        if len(X_time) > 0:
            last_sequence = X_time[-1:]
            
            # 다음 달 단어-점수 데이터 로드
            if args.next_month:
                # 지정된 월의 데이터 사용
                from src.data_loader import DataLoader
                input_data = pipeline.data_loader.load_input_data(month=args.next_month)
                if args.next_month in input_data:
                    word_scores = input_data[args.next_month]
                    word_columns = [col.replace('word_score_', '') for col in data_info['word_columns']]
                    feature_vector = [word_scores.get(word, 0.0) for word in word_columns]
                    last_word_features = np.array([feature_vector])
                else:
                    print(f"경고: {args.next_month}의 단어-점수 데이터가 없습니다.")
                    last_word_features = X_word[-1:]
            else:
                # 자동으로 다음 달 데이터 로드
                last_word_features = pipeline._load_next_month_word_features(data_info['word_columns'])
                if last_word_features is None:
                    print("경고: 다음 달 단어-점수 데이터가 없습니다. 마지막 달 데이터를 사용합니다.")
                    last_word_features = X_word[-1:]
            
            prediction = pipeline.predict_next_month(last_sequence, last_word_features)
            print(f"\n다음 달 예측 매출: {prediction:,.2f}")
    else:
        # 전체 파이프라인 실행
        result = pipeline.run_full_pipeline(retrain=args.retrain)
        
        if result['status'] == 'success':
            print("\n파이프라인 실행 완료!")
            print(f"예측 매출: {result['prediction']:,.2f}")
        else:
            print(f"\n파이프라인 실행 실패: {result.get('message', result['status'])}")
            sys.exit(1)


if __name__ == '__main__':
    main()

