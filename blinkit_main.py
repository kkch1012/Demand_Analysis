"""
Blinkit 수요 예측 메인 실행 스크립트
Blinkit 데이터를 활용한 월별 매출 예측
"""

import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

from src.blinkit.pipeline import BlinkitPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Blinkit 수요 예측 파이프라인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python blinkit_main.py                    # 전체 파이프라인 실행
  python blinkit_main.py --retrain          # 강제 재학습
  python blinkit_main.py --predict-only     # 예측만 수행
  python blinkit_main.py --visualize        # 결과 시각화 포함
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/blinkit_config.yaml',
        help='설정 파일 경로 (기본: config/blinkit_config.yaml)'
    )
    
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='모델 강제 재학습'
    )
    
    parser.add_argument(
        '--predict-only',
        action='store_true',
        help='예측만 수행 (학습 생략, 기존 모델 사용)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='결과 시각화 포함'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='출력 상세도 (0: 최소, 1: 기본, 2: 상세)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("     🛒 Blinkit 수요 예측 시스템")
    print("=" * 60)
    
    # 파이프라인 초기화
    try:
        pipeline = BlinkitPipeline(config_path=args.config)
    except Exception as e:
        print(f"\n❌ 설정 파일 로드 실패: {e}")
        sys.exit(1)
    
    if args.predict_only:
        # 예측만 수행
        print("\n📊 예측 모드 실행...")
        
        # 데이터 로딩
        pipeline.load_and_prepare_data()
        pipeline.prepare_training_data()
        
        # 모델 로드
        if not pipeline.load_model():
            print("\n❌ 오류: 학습된 모델이 없습니다.")
            print("   먼저 'python blinkit_main.py'로 학습을 실행하세요.")
            sys.exit(1)
        
        # 예측
        result = pipeline.predict_next_period()
        
        print("\n" + "=" * 60)
        print("     📈 예측 완료")
        print("=" * 60)
        print(f"\n  다음 달 예상 매출: {result['predicted_value']:,.0f}원")
        print(f"  전월 대비 변화: {result['change_percent']:+.1f}%")
        
    else:
        # 전체 파이프라인 실행
        result = pipeline.run_full_pipeline(
            retrain=args.retrain,
            verbose=args.verbose
        )
        
        if result['status'] == 'success':
            print("\n" + "=" * 60)
            print("     ✅ 파이프라인 실행 성공")
            print("=" * 60)
            
            print(f"\n📊 평가 지표:")
            for name, value in result['evaluation'].items():
                print(f"   - {name}: {value:,.2f}")
            
            print(f"\n📈 예측 결과:")
            pred = result['prediction']
            print(f"   - 마지막 달 ({pred['last_month']}): {pred['last_value']:,.0f}원")
            print(f"   - 다음 달 예측: {pred['predicted_value']:,.0f}원")
            print(f"   - 변화율: {pred['change_percent']:+.1f}%")
            
            print(f"\n⏱️ 실행 시간: {result['execution_time']}")
            
            # 시각화
            if args.visualize:
                print("\n📊 결과 시각화 중...")
                try:
                    eval_result = pipeline.evaluate_model()
                    pipeline.visualize_results(eval_result)
                    
                    # 학습 히스토리
                    if pipeline.model and pipeline.model.history:
                        pipeline.model.plot_training_history()
                except Exception as e:
                    print(f"   시각화 오류: {e}")
        
        elif result['status'] == 'insufficient_data':
            print(f"\n⚠️ 경고: {result['message']}")
            sys.exit(1)
        
        else:
            print(f"\n❌ 파이프라인 실행 실패")
            print(f"   오류: {result.get('message', 'Unknown error')}")
            if 'traceback' in result:
                print(f"\n상세 오류:\n{result['traceback']}")
            sys.exit(1)


if __name__ == '__main__':
    main()

