"""
Blinkit 수요 예측 - 일간 vs 주간 비교 스크립트
두 가지 집계 방식으로 모델을 학습하고 성능을 비교합니다.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.blinkit.pipeline import BlinkitPipeline


def run_comparison(freqs=['daily', 'weekly'], verbose=0):
    """
    일간/주간 집계로 모델을 학습하고 비교
    
    Args:
        freqs: 비교할 집계 주기 리스트
        verbose: 출력 상세도
        
    Returns:
        비교 결과 딕셔너리
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("       🔬 Blinkit 수요 예측 - 집계 방식 비교 분석")
    print("=" * 70)
    
    for freq in freqs:
        freq_name = {'daily': '일간', 'weekly': '주간', 'monthly': '월간'}[freq]
        
        print(f"\n\n{'=' * 70}")
        print(f"       📊 [{freq_name} 집계] 모델 학습 시작")
        print("=" * 70)
        
        try:
            # 파이프라인 초기화
            pipeline = BlinkitPipeline(freq=freq)
            
            # 시퀀스 길이 조정
            if freq == 'daily':
                pipeline.sequence_length = 14  # 2주
                pipeline.config['model']['time_series']['sequence_length'] = 14
            elif freq == 'weekly':
                pipeline.sequence_length = 8   # 8주 (약 2달)
                pipeline.config['model']['time_series']['sequence_length'] = 8
            
            # 데이터 로딩 및 전처리
            pipeline.load_and_prepare_data()
            
            # 학습 데이터 준비
            pipeline.prepare_training_data()
            
            # 데이터 양 확인
            n_samples = len(pipeline.y)
            print(f"\n학습 샘플 수: {n_samples}개")
            
            if n_samples < 10:
                print(f"⚠️ 경고: 데이터가 너무 적습니다 ({n_samples}개). 스킵합니다.")
                results[freq] = {'status': 'insufficient_data', 'n_samples': n_samples}
                continue
            
            # 모델 학습
            training_result = pipeline.train_model(verbose=verbose)
            
            # 모델 평가
            eval_result = pipeline.evaluate_model()
            
            # 예측
            prediction = pipeline.predict_next_period()
            
            # 결과 저장
            results[freq] = {
                'status': 'success',
                'freq_name': freq_name,
                'n_samples': n_samples,
                'training': training_result,
                'metrics': eval_result['metrics'],
                'prediction': prediction,
                'y_true': eval_result['y_true'],
                'y_pred': eval_result['y_pred'],
                'pipeline': pipeline
            }
            
            # 모델 저장
            pipeline.save_model()
            
        except Exception as e:
            import traceback
            print(f"\n❌ 오류 발생: {e}")
            results[freq] = {
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
    
    return results


def print_comparison_table(results):
    """비교 결과 테이블 출력"""
    
    print("\n\n" + "=" * 70)
    print("       📈 성능 비교 결과")
    print("=" * 70)
    
    # 성공한 결과만 필터링
    success_results = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if not success_results:
        print("\n⚠️ 비교할 수 있는 결과가 없습니다.")
        return
    
    # 테이블 헤더
    print(f"\n{'집계방식':<10} {'샘플수':>10} {'MAE':>15} {'RMSE':>15} {'R2':>10} {'MAPE(%)':>10}")
    print("-" * 70)
    
    best_r2 = -float('inf')
    best_freq = None
    
    for freq, result in success_results.items():
        metrics = result['metrics']
        print(f"{result['freq_name']:<10} {result['n_samples']:>10} "
              f"{metrics['MAE']:>15,.2f} {metrics['RMSE']:>15,.2f} "
              f"{metrics['R2']:>10.4f} {metrics['MAPE']:>10.2f}")
        
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_freq = freq
    
    print("-" * 70)
    
    if best_freq:
        print(f"\n🏆 최고 성능: {success_results[best_freq]['freq_name']} 집계 (R2 = {best_r2:.4f})")
    
    # 예측 결과 비교
    print("\n\n📊 예측 결과 비교:")
    print("-" * 70)
    
    for freq, result in success_results.items():
        pred = result['prediction']
        print(f"\n[{result['freq_name']}]")
        print(f"  마지막 기간: {pred['last_period']}")
        print(f"  마지막 값: {pred['last_value']:,.2f}")
        print(f"  다음 기간 예측: {pred['predicted_value']:,.2f}")
        print(f"  변화율: {pred['change_percent']:+.2f}%")


def plot_comparison(results, save_path=None):
    """비교 결과 시각화"""
    
    success_results = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if len(success_results) < 1:
        print("시각화할 결과가 없습니다.")
        return
    
    n_plots = len(success_results)
    fig, axes = plt.subplots(2, n_plots, figsize=(7*n_plots, 10))
    
    if n_plots == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (freq, result) in enumerate(success_results.items()):
        y_true = result['y_true']
        y_pred = result['y_pred']
        metrics = result['metrics']
        
        # 실제 vs 예측
        ax1 = axes[0, idx]
        ax1.plot(y_true, 'b-o', label='실제', markersize=3, alpha=0.7)
        ax1.plot(y_pred, 'r--s', label='예측', markersize=3, alpha=0.7)
        ax1.set_title(f'{result["freq_name"]} - 실제 vs 예측\n(R2={metrics["R2"]:.3f}, MAPE={metrics["MAPE"]:.1f}%)')
        ax1.set_xlabel('샘플')
        ax1.set_ylabel('매출')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 산점도
        ax2 = axes[1, idx]
        ax2.scatter(y_true, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        ax2.set_title(f'{result["freq_name"]} - 예측 정확도')
        ax2.set_xlabel('실제 값')
        ax2.set_ylabel('예측 값')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n시각화 저장: {save_path}")
    
    plt.show()


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Blinkit 일간/주간 집계 비교')
    parser.add_argument('--daily', action='store_true', help='일간 집계 포함')
    parser.add_argument('--weekly', action='store_true', help='주간 집계 포함')
    parser.add_argument('--monthly', action='store_true', help='월간 집계 포함')
    parser.add_argument('--all', action='store_true', help='모든 집계 방식')
    parser.add_argument('--verbose', type=int, default=0, help='출력 상세도')
    parser.add_argument('--no-plot', action='store_true', help='시각화 생략')
    
    args = parser.parse_args()
    
    # 집계 방식 선택
    if args.all:
        freqs = ['daily', 'weekly', 'monthly']
    else:
        freqs = []
        if args.daily:
            freqs.append('daily')
        if args.weekly:
            freqs.append('weekly')
        if args.monthly:
            freqs.append('monthly')
        
        # 기본값: 일간, 주간
        if not freqs:
            freqs = ['daily', 'weekly']
    
    # 비교 실행
    results = run_comparison(freqs, verbose=args.verbose)
    
    # 결과 출력
    print_comparison_table(results)
    
    # 시각화
    if not args.no_plot:
        try:
            plot_comparison(
                results, 
                save_path=f'predictions/comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )
        except Exception as e:
            print(f"\n시각화 오류: {e}")
    
    print("\n✅ 비교 완료!")
    
    return results


if __name__ == '__main__':
    main()

