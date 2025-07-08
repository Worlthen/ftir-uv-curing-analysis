#!/usr/bin/env python3
"""
FTIR分析系统测试脚本
测试核心分析功能而不启动GUI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from system_analysis import FTIRAnalyzer

def test_analysis():
    """测试分析功能"""
    print("="*60)
    print("FTIR光谱分析系统测试")
    print("="*60)
    
    # 创建分析器
    analyzer = FTIRAnalyzer()
    
    # 加载数据
    print("\n1. 加载数据...")
    if not analyzer.load_data('all_spectra.csv'):
        print("数据加载失败！")
        return
    
    # 执行分析
    print("\n2. 执行综合分析...")
    try:
        # 选择部分时间点进行快速测试
        selected_times = [0, 2, 4, 8, 16]
        selected_wavenumbers = (1000, 2000)  # 选择部分波数范围
        
        results = analyzer.perform_comprehensive_analysis(
            baseline_method='als',
            normalize_method='max',
            selected_times=selected_times,
            selected_wavenumbers=selected_wavenumbers
        )
        
        print("分析完成！")
        
    except Exception as e:
        print(f"分析失败: {e}")
        return
    
    # 显示结果
    print("\n3. 分析结果概览:")
    print("-"*40)
    
    if 'key_analysis' in results:
        print("\n关键波数区域分析结果:")
        for region, data in results['key_analysis'].items():
            if data['best_fit']:
                fit = data['best_fit']
                print(f"\n{region}:")
                print(f"  - 波数范围: {data['wavenumber_range'][0]}-{data['wavenumber_range'][1]} cm⁻¹")
                print(f"  - 最佳模型: {fit['model']}")
                print(f"  - 速率常数: {fit['k']:.2e} s⁻¹")
                print(f"  - 拟合优度 R²: {fit['r2']:.4f}")
                print(f"  - 半衰期: {fit['half_life']:.2f} 秒")
    
    if 'pca_results' in results:
        pca = results['pca_results']
        print(f"\n主成分分析:")
        for i, var in enumerate(pca['explained_variance_ratio']):
            print(f"  - PC{i+1}: {var*100:.1f}% 方差解释")
        print(f"  - 累积方差解释: {pca['cumulative_variance'][-1]*100:.1f}%")
    
    # 生成报告
    print("\n4. 生成分析报告...")
    analyzer.generate_report()
    
    # 创建简单的可视化
    print("\n5. 生成可视化图表...")
    create_simple_plots(analyzer, results)
    
    print("\n测试完成！")

def create_simple_plots(analyzer, results):
    """创建简单的可视化图表"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('FTIR光谱分析结果', fontsize=16)
        
        # 1. 光谱演变图
        ax1 = axes[0, 0]
        processed_data = results['processed_data']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (time, spectrum) in enumerate(processed_data.items()):
            if i < len(colors):
                ax1.plot(spectrum['Wavenumber'], spectrum['Absorbance'], 
                        color=colors[i], label=f'{time}s', linewidth=1.5)
        
        ax1.set_xlabel('波数 (cm⁻¹)')
        ax1.set_ylabel('归一化吸光度')
        ax1.set_title('光谱演变')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 关键波数时间序列
        ax2 = axes[0, 1]
        if 'key_analysis' in results:
            for region, data in results['key_analysis'].items():
                if 'time_series' in data:
                    ts = data['time_series']
                    ax2.plot(ts['time'], ts['absorbance'], 'o-', label=region, linewidth=2)
        
        ax2.set_xlabel('曝光时间 (s)')
        ax2.set_ylabel('平均吸光度')
        ax2.set_title('关键波数区域时间序列')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 差谱分析
        ax3 = axes[1, 0]
        if 'difference_spectra' in results:
            diff_spectra = results['difference_spectra']
            for i, (time, diff_data) in enumerate(diff_spectra.items()):
                if i < len(colors):
                    ax3.plot(diff_data['Wavenumber'], diff_data['Difference'], 
                            color=colors[i], label=f'{time}s', linewidth=1.5)
        
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('波数 (cm⁻¹)')
        ax3.set_ylabel('吸光度差值')
        ax3.set_title('差谱分析')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. PCA得分图
        ax4 = axes[1, 1]
        if 'pca_results' in results:
            pca = results['pca_results']
            scores = pca['scores']
            times = results['analysis_times']
            
            ax4.scatter(scores[:, 0], scores[:, 1], c=times, s=100, cmap='viridis')
            
            # 添加时间标签
            for i, time in enumerate(times):
                ax4.annotate(f'{time}s', (scores[i, 0], scores[i, 1]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax4.set_xlabel(f'PC1 ({pca["explained_variance_ratio"][0]*100:.1f}%)')
            ax4.set_ylabel(f'PC2 ({pca["explained_variance_ratio"][1]*100:.1f}%)')
            ax4.set_title('PCA得分图')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ftir_analysis_results.png', dpi=300, bbox_inches='tight')
        print("图表已保存为: ftir_analysis_results.png")
        
    except Exception as e:
        print(f"绘图失败: {e}")

if __name__ == "__main__":
    test_analysis()
