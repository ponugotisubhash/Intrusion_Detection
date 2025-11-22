"""
Comprehensive Evaluation and Visualization Module
Handles model comparison, performance analysis, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-GUI environment
plt.switch_backend('Agg')

class ModelEvaluator:
    def __init__(self, results_dir="/home/user/intrusion_detection_project/results"):
        """
        Initialize model evaluator
        
        Args:
            results_dir: Directory to save results and plots
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set style for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_performance_comparison(self, baseline_results, cnn_results, 
                                   baseline_test_results=None, cnn_test_results=None):
        """
        Create comprehensive performance comparison between models
        
        Args:
            baseline_results: Results from baseline models
            cnn_results: Results from CNN model
            baseline_test_results: Test results from baseline models
            cnn_test_results: Test results from CNN model
            
        Returns:
            Comparison dataframe and visualizations
        """
        print(" Creating performance comparison...")
        
        comparison_data = []
        
        # Add baseline model results
        for model_name, results in baseline_results.items():
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Type': 'Traditional ML',
                'Val_Accuracy': results['metrics']['accuracy'],
                'Val_F1_Score': results['metrics']['f1_score'],
                'Val_AUC_ROC': results['metrics']['auc_roc'],
                'Val_Precision': results['metrics']['precision'],
                'Val_Recall': results['metrics']['recall'],
                'Training_Time_s': results['training_time'],
                'Inference_Time_ms': results['inference_time_ms'],
                'Memory_Usage_MB': results['memory_usage_mb']
            }
            
            # Add test results if available
            if baseline_test_results and model_name in baseline_test_results:
                test_metrics = baseline_test_results[model_name]['metrics']
                row.update({
                    'Test_Accuracy': test_metrics['accuracy'],
                    'Test_F1_Score': test_metrics['f1_score'],
                    'Test_AUC_ROC': test_metrics['auc_roc'],
                    'Test_Precision': test_metrics['precision'],
                    'Test_Recall': test_metrics['recall']
                })
            
            comparison_data.append(row)
        
        # Add CNN results
        if cnn_results:
            row = {
                'Model': 'Lightweight CNN',
                'Type': 'Deep Learning',
                'Val_Accuracy': cnn_results['metrics']['accuracy'],
                'Val_F1_Score': cnn_results['metrics']['f1_score'],
                'Val_AUC_ROC': cnn_results['metrics']['auc_roc'],
                'Val_Precision': cnn_results['metrics']['precision'],
                'Val_Recall': cnn_results['metrics']['recall'],
                'Training_Time_s': cnn_results['training_time'],
                'Inference_Time_ms': cnn_results['inference_time_ms'],
                'Memory_Usage_MB': cnn_results['memory_usage_mb']
            }
            
            # Add CNN test results if available
            if cnn_test_results:
                test_metrics = cnn_test_results['metrics']
                row.update({
                    'Test_Accuracy': test_metrics['accuracy'],
                    'Test_F1_Score': test_metrics['f1_score'],
                    'Test_AUC_ROC': test_metrics['auc_roc'],
                    'Test_Precision': test_metrics['precision'],
                    'Test_Recall': test_metrics['recall']
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_path = os.path.join(self.results_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f" Saved comparison table to {comparison_path}")
        
        return comparison_df
    
    def plot_performance_metrics(self, comparison_df):
        """
        Create visualization of performance metrics
        
        Args:
            comparison_df: DataFrame with model comparison results
        """
        print(" Creating performance visualizations...")
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Metrics to plot
        metrics = ['Val_Accuracy', 'Val_F1_Score', 'Val_AUC_ROC', 
                  'Training_Time_s', 'Inference_Time_ms', 'Memory_Usage_MB']
        
        titles = ['Validation Accuracy', 'Validation F1-Score', 'Validation AUC-ROC',
                 'Training Time (seconds)', 'Inference Time (ms)', 'Memory Usage (MB)']
        
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Create bar plot
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=[colors[i % len(colors)] for i in range(len(comparison_df))])
            
            # Customize plot
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' '))
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, "performance_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved performance plot to {plot_path}")
    
    def plot_roc_curves(self, models_data, test_labels):
        """
        Plot ROC curves for all models
        
        Args:
            models_data: Dictionary with model predictions and probabilities
            test_labels: True test labels
        """
        print(" Creating ROC curve comparison...")
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, (model_name, data) in enumerate(models_data.items()):
            if 'probabilities' in data:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(test_labels, data['probabilities'])
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, 
                        color=colors[idx % len(colors)],
                        lw=2, 
                        label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curves Comparison', fontweight='bold', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        roc_path = os.path.join(self.results_dir, "roc_curves.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved ROC curves to {roc_path}")
    
    def plot_confusion_matrices(self, models_data, test_labels):
        """
        Plot confusion matrices for all models
        
        Args:
            models_data: Dictionary with model predictions
            test_labels: True test labels
        """
        print(" Creating confusion matrices...")
        
        num_models = len(models_data)
        cols = min(3, num_models)
        rows = (num_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if num_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if num_models > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, data) in enumerate(models_data.items()):
            if 'predictions' in data:
                # Calculate confusion matrix
                cm = confusion_matrix(test_labels, data['predictions'])
                
                # Plot confusion matrix
                ax = axes[idx] if num_models > 1 else axes[0]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Benign', 'Attack'],
                           yticklabels=['Benign', 'Attack'])
                
                ax.set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Predicted Label', fontweight='bold')
                ax.set_ylabel('True Label', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(num_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        cm_path = os.path.join(self.results_dir, "confusion_matrices.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved confusion matrices to {cm_path}")
    
    def plot_efficiency_analysis(self, comparison_df):
        """
        Create efficiency analysis plots (accuracy vs. inference time, memory usage)
        
        Args:
            comparison_df: DataFrame with model comparison results
        """
        print(" Creating efficiency analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs Inference Time
        ax1 = axes[0]
        colors = ['blue', 'red', 'green', 'orange']
        for idx, (_, row) in enumerate(comparison_df.iterrows()):
            ax1.scatter(row['Inference_Time_ms'], row['Val_Accuracy'], 
                       s=100, color=colors[idx % len(colors)], 
                       label=row['Model'], alpha=0.7)
        
        ax1.set_xlabel('Inference Time (ms)', fontweight='bold')
        ax1.set_ylabel('Validation Accuracy', fontweight='bold')
        ax1.set_title('Accuracy vs. Inference Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy vs Memory Usage
        ax2 = axes[1]
        for idx, (_, row) in enumerate(comparison_df.iterrows()):
            ax2.scatter(row['Memory_Usage_MB'], row['Val_Accuracy'], 
                       s=100, color=colors[idx % len(colors)], 
                       label=row['Model'], alpha=0.7)
        
        ax2.set_xlabel('Memory Usage (MB)', fontweight='bold')
        ax2.set_ylabel('Validation Accuracy', fontweight='bold')
        ax2.set_title('Accuracy vs. Memory Usage', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        eff_path = os.path.join(self.results_dir, "efficiency_analysis.png")
        plt.savefig(eff_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved efficiency analysis to {eff_path}")
    
    def create_detailed_report(self, comparison_df, models_data=None, test_labels=None):
        """
        Create a detailed evaluation report
        
        Args:
            comparison_df: DataFrame with model comparison results
            models_data: Dictionary with model predictions (optional)
            test_labels: True test labels (optional)
        """
        print(" Creating detailed evaluation report...")
        
        report_path = os.path.join(self.results_dir, "evaluation_report.txt")
        
        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("INTRUSION DETECTION MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            # Find best performing model
            best_accuracy = comparison_df.loc[comparison_df['Val_Accuracy'].idxmax()]
            best_f1 = comparison_df.loc[comparison_df['Val_F1_Score'].idxmax()]
            fastest_inference = comparison_df.loc[comparison_df['Inference_Time_ms'].idxmin()]
            
            f.write(f"Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Val_Accuracy']:.4f})\n")
            f.write(f"Best F1-Score: {best_f1['Model']} ({best_f1['Val_F1_Score']:.4f})\n")
            f.write(f"Fastest Inference: {fastest_inference['Model']} ({fastest_inference['Inference_Time_ms']:.2f}ms)\n\n")
            
            f.write("DETAILED RESULTS\n")
            f.write("-" * 40 + "\n")
            
            for _, row in comparison_df.iterrows():
                f.write(f"\n{row['Model'].upper()}\n")
                f.write("-" * len(row['Model']) + "\n")
                f.write(f"Type: {row['Type']}\n")
                f.write(f"Validation Accuracy: {row['Val_Accuracy']:.4f}\n")
                f.write(f"Validation F1-Score: {row['Val_F1_Score']:.4f}\n")
                f.write(f"Validation AUC-ROC: {row['Val_AUC_ROC']:.4f}\n")
                f.write(f"Training Time: {row['Training_Time_s']:.2f}s\n")
                f.write(f"Inference Time: {row['Inference_Time_ms']:.2f}ms\n")
                f.write(f"Memory Usage: {row['Memory_Usage_MB']:.2f}MB\n")
                
                # Add test results if available
                if 'Test_Accuracy' in row and pd.notna(row['Test_Accuracy']):
                    f.write(f"Test Accuracy: {row['Test_Accuracy']:.4f}\n")
                    f.write(f"Test F1-Score: {row['Test_F1_Score']:.4f}\n")
                    f.write(f"Test AUC-ROC: {row['Test_AUC_ROC']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("HARDWARE FEASIBILITY ANALYSIS\n")
            f.write("=" * 80 + "\n")
            
            # Check which models meet the requirements
            f.write("\nREQUIREMENT COMPLIANCE:\n")
            f.write("- Target: >90% Accuracy, ≤10ms Inference, ≤4GB Memory\n\n")
            
            for _, row in comparison_df.iterrows():
                meets_accuracy = row['Val_Accuracy'] >= 0.90
                meets_speed = row['Inference_Time_ms'] <= 10.0
                meets_memory = row['Memory_Usage_MB'] <= 4000  # 4GB in MB
                
                f.write(f"{row['Model']}:\n")
                f.write(f"  Accuracy ≥90%: {'✓' if meets_accuracy else '✗'} ({row['Val_Accuracy']:.1%})\n")
                f.write(f"  Speed ≤10ms: {'✓' if meets_speed else '✗'} ({row['Inference_Time_ms']:.2f}ms)\n")
                f.write(f"  Memory ≤4GB: {'✓' if meets_memory else '✗'} ({row['Memory_Usage_MB']:.1f}MB)\n")
                f.write(f"  Overall: {'PASS' if all([meets_accuracy, meets_speed, meets_memory]) else 'FAIL'}\n\n")
            
            # Detailed classification reports
            if models_data and test_labels is not None:
                f.write("\nDETAILED CLASSIFICATION REPORTS\n")
                f.write("-" * 40 + "\n")
                
                for model_name, data in models_data.items():
                    if 'predictions' in data:
                        f.write(f"\n{model_name.upper().replace('_', ' ')}\n")
                        f.write("-" * len(model_name) + "\n")
                        
                        report = classification_report(
                            test_labels, data['predictions'],
                            target_names=['Benign', 'Attack'],
                            digits=4
                        )
                        f.write(report + "\n")
        
        print(f" Saved detailed report to {report_path}")
        return report_path
    
    def benchmark_real_time_performance(self, models, X_test, batch_sizes=[1, 10, 100, 1000]):
        """
        Benchmark real-time performance with different batch sizes
        
        Args:
            models: Dictionary of trained models
            X_test: Test data for benchmarking
            batch_sizes: List of batch sizes to test
            
        Returns:
            Benchmarking results
        """
        print(" Benchmarking real-time performance...")
        
        benchmark_results = []
        
        for model_name, model in models.items():
            print(f"   Testing {model_name}...")
            
            for batch_size in batch_sizes:
                # Select batch
                batch_data = X_test[:batch_size]
                
                # Measure inference time
                times = []
                for _ in range(10):  # Multiple runs for stability
                    start_time = time.time()
                    
                    # Make predictions
                    if hasattr(model, 'predict_proba'):
                        _ = model.predict_proba(batch_data)
                    elif hasattr(model, 'predict'):
                        _ = model.predict(batch_data)
                    
                    times.append((time.time() - start_time) * 1000)  # Convert to ms
                
                avg_time = np.mean(times)
                per_sample_time = avg_time / batch_size
                
                benchmark_results.append({
                    'model': model_name,
                    'batch_size': batch_size,
                    'total_time_ms': avg_time,
                    'per_sample_ms': per_sample_time,
                    'throughput_samples_per_sec': 1000 / per_sample_time
                })
        
        benchmark_df = pd.DataFrame(benchmark_results)
        
        # Save results
        benchmark_path = os.path.join(self.results_dir, "realtime_benchmark.csv")
        benchmark_df.to_csv(benchmark_path, index=False)
        
        print(f" Saved benchmark results to {benchmark_path}")
        return benchmark_df

def main():
    """
    Demonstration of comprehensive evaluation
    """
    # This would typically be called after training all models
    print(" Comprehensive Model Evaluation Demo")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Create sample results for demonstration
    sample_results = {
        'random_forest': {
            'metrics': {'accuracy': 0.9234, 'f1_score': 0.8956, 'auc_roc': 0.9567, 
                       'precision': 0.9123, 'recall': 0.8789},
            'training_time': 12.45,
            'inference_time_ms': 2.34,
            'memory_usage_mb': 145.67
        },
        'xgboost': {
            'metrics': {'accuracy': 0.9456, 'f1_score': 0.9234, 'auc_roc': 0.9678, 
                       'precision': 0.9345, 'recall': 0.9123},
            'training_time': 23.67,
            'inference_time_ms': 1.89,
            'memory_usage_mb': 234.56
        }
    }
    
    cnn_results = {
        'metrics': {'accuracy': 0.9345, 'f1_score': 0.9123, 'auc_roc': 0.9645, 
                   'precision': 0.9234, 'recall': 0.9012},
        'training_time': 45.67,
        'inference_time_ms': 3.45,
        'memory_usage_mb': 67.89
    }
    
    # Create comparison
    comparison_df = evaluator.create_performance_comparison(sample_results, cnn_results)
    
    print("\n Model Comparison Results:")
    print(comparison_df.round(4))
    
    # Create visualizations
    evaluator.plot_performance_metrics(comparison_df)
    evaluator.plot_efficiency_analysis(comparison_df)
    
    # Create detailed report
    report_path = evaluator.create_detailed_report(comparison_df)
    
    print(f"\n Comprehensive evaluation completed!")
    print(f" Results saved in: /home/user/intrusion_detection_project/results/")
    
    return evaluator, comparison_df

if __name__ == "__main__":
    main()