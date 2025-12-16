"""
메인 실행 스크립트
자동화 파이프라인 실행
"""

import argparse
import sys
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

