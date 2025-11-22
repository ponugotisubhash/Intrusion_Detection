

import os
import sys
import time
import warnings
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from data_loader import CICIDS2017DataLoader
from baseline_models import BaselineModels
from lightweight_cnn import LightweightCNN
from evaluation import ModelEvaluator

warnings.filterwarnings('ignore')

class IntrustionDetectionPipeline:
    def __init__(self, project_dir="/intrusion_detection_project"):
        """
        Initialize the intrusion detection pipeline
        
        Args:
            project_dir: Root directory of the project
        """
        self.project_dir = project_dir
        self.data_dir = os.path.join(project_dir, "data")
        self.models_dir = os.path.join(project_dir, "models")
        self.results_dir = os.path.join(project_dir, "results")
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.baseline_models = None
        self.cnn_model = None
        self.evaluator = None
        
        # Store results
        self.processed_data = None
        self.baseline_results = None
        self.cnn_results = None
        self.test_results = {}
        self.comparison_results = None
    
    def run_complete_pipeline(self, sample_size=10000, quick_run=False):
        """
        Run the complete machine learning pipeline
        
        Args:
            sample_size: Number of samples to generate for demonstration
            quick_run: If True, use reduced parameters for faster execution
        """
        print(" STARTING INTRUSION DETECTION PIPELINE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Data Loading and Preprocessing
        print("\n STEP 1: DATA LOADING AND PREPROCESSING")
        print("-" * 50)
        self._load_and_preprocess_data(sample_size)
        
        # Step 2: Train Baseline Models
        print("\n STEP 2: TRAINING BASELINE MODELS")
        print("-" * 50)
        self._train_baseline_models(quick_run=quick_run)
        
        # Step 3: Train CNN Model
        print("\n STEP 3: TRAINING LIGHTWEIGHT CNN")
        print("-" * 50)
        self._train_cnn_model(quick_run=quick_run)
        
        # Step 4: Evaluate All Models
        print("\n STEP 4: MODEL EVALUATION")
        print("-" * 50)
        self._evaluate_all_models()
        
        # Step 5: Comprehensive Analysis
        print("\n STEP 5: COMPREHENSIVE ANALYSIS")
        print("-" * 50)
        self._comprehensive_analysis()
        
        # Step 6: Generate Final Report
        print("\n STEP 6: GENERATING FINAL REPORT")
        print("-" * 50)
        self._generate_final_report()
        
        total_time = time.time() - start_time
        print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
        print(f"  Total Execution Time: {total_time:.2f} seconds")
        print(f" Results saved in: {self.results_dir}")
        
        return self._get_pipeline_summary()
    
    def _load_and_preprocess_data(self, sample_size):
        """Load and preprocess the dataset"""
        print(" Initializing data loader...")
        self.data_loader = CICIDS2017DataLoader()
        
        print(f" Generating synthetic CICIDS2017 dataset ({sample_size:,} samples)...")
        df = self.data_loader.generate_synthetic_data(n_samples=sample_size)
        
        print("📋 Dataset Information:")
        info = self.data_loader.get_data_info(df)
        print(f"   Shape: {info['shape']}")
        print(f"   Memory Usage: {info['memory_usage']:.2f} MB")
        print(f"   Missing Values: {info['missing_values']}")
        
        print("\n Label Distribution:")
        for label, count in info['label_distribution'].items():
            percentage = info['label_percentages'][label]
            print(f"   {label}: {count:,} ({percentage:.2f}%)")
        
        print("\n🔧 Preprocessing data (scaling, SMOTE, train/val/test split)...")
        self.processed_data = self.data_loader.preprocess_data(df, apply_smote=True)
        
        print(" Data preprocessing completed!")
        print(f"   Training set: {self.processed_data['X_train'].shape}")
        print(f"   Validation set: {self.processed_data['X_val'].shape}")
        print(f"   Test set: {self.processed_data['X_test'].shape}")
    
    def _train_baseline_models(self, quick_run=False):
        """Train baseline machine learning models"""
        self.baseline_models = BaselineModels()
        
        # Extract data
        X_train = self.processed_data['X_train']
        X_val = self.processed_data['X_val']
        y_train = self.processed_data['y_train']
        y_val = self.processed_data['y_val']
        
        # Adjust parameters for quick run
        n_estimators = 50 if quick_run else 100
        
        print(f" Training Random Forest (n_estimators={n_estimators})...")
        rf_model, rf_results = self.baseline_models.train_random_forest(
            X_train, y_train, X_val, y_val, n_estimators=n_estimators
        )
        
        print(f"\n Training XGBoost/GradientBoosting (n_estimators={n_estimators})...")
        xgb_model, xgb_results = self.baseline_models.train_xgboost(
            X_train, y_train, X_val, y_val, n_estimators=n_estimators
        )
        
        self.baseline_results = self.baseline_models.results
        
        print(f"\n Saving baseline models...")
        self.baseline_models.save_models(self.models_dir)
        
        print("Baseline model training completed!")
    
    def _train_cnn_model(self, quick_run=False):
        """Train lightweight CNN model"""
        self.cnn_model = LightweightCNN()
        
        # Extract data
        X_train = self.processed_data['X_train']
        X_val = self.processed_data['X_val']
        y_train = self.processed_data['y_train']
        y_val = self.processed_data['y_val']
        
        # Adjust parameters for quick run
        epochs = 20 if quick_run else 50
        
        print(f" Training Lightweight CNN (epochs={epochs})...")
        history, cnn_results = self.cnn_model.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            depth=4,
            initial_filters=64,
            batch_size=512
        )
        
        self.cnn_results = cnn_results
        
        print(f"\n Saving CNN model...")
        cnn_model_path = os.path.join(self.models_dir, "lightweight_cnn")
        self.cnn_model.save_model(cnn_model_path)
        
        print(" CNN model training completed!")
    
    def _evaluate_all_models(self):
        """Evaluate all trained models on test set"""
        X_test = self.processed_data['X_test']
        y_test = self.processed_data['y_test']
        
        print(" Evaluating baseline models on test set...")
        baseline_test_results = self.baseline_models.evaluate_on_test(X_test, y_test)
        self.test_results.update(baseline_test_results)
        
        if self.cnn_model.model is not None:
            print("\n Evaluating CNN model on test set...")
            cnn_test_results = self.cnn_model.evaluate(X_test, y_test)
            self.test_results['lightweight_cnn'] = cnn_test_results
        
        print(" Model evaluation completed!")
    
    def _comprehensive_analysis(self):
        """Perform comprehensive model analysis"""
        print(" Initializing comprehensive evaluator...")
        self.evaluator = ModelEvaluator(self.results_dir)
        
        # Create performance comparison
        cnn_results_for_comparison = self.cnn_results if self.cnn_model.model is not None else None
        cnn_test_for_comparison = self.test_results.get('lightweight_cnn', None)
        
        baseline_test_results = {k: v for k, v in self.test_results.items() 
                               if k != 'lightweight_cnn'}
        
        print(" Creating performance comparison...")
        self.comparison_results = self.evaluator.create_performance_comparison(
            self.baseline_results,
            cnn_results_for_comparison,
            baseline_test_results,
            cnn_test_for_comparison
        )
        
        print("\n Generating visualizations...")
        # Performance metrics plots
        self.evaluator.plot_performance_metrics(self.comparison_results)
        
        # Efficiency analysis
        self.evaluator.plot_efficiency_analysis(self.comparison_results)
        
        # ROC curves (if we have probabilities)
        models_with_probs = {k: v for k, v in self.test_results.items() 
                           if 'probabilities' in v}
        if models_with_probs:
            y_test = self.processed_data['y_test']
            self.evaluator.plot_roc_curves(models_with_probs, y_test)
            self.evaluator.plot_confusion_matrices(models_with_probs, y_test)
        
        print("Comprehensive analysis completed!")
    
    def _generate_final_report(self):
        """Generate final comprehensive report"""
        print(" Generating detailed evaluation report...")
        
        # Create detailed report
        models_data = self.test_results if self.test_results else None
        y_test = self.processed_data['y_test'] if self.processed_data else None
        
        report_path = self.evaluator.create_detailed_report(
            self.comparison_results, 
            models_data, 
            y_test
        )
        
        # Real-time benchmarking if models available
        if hasattr(self.baseline_models, 'models') and self.baseline_models.models:
            print("\n Running real-time performance benchmarking...")
            models_for_benchmark = self.baseline_models.models.copy()
            if self.cnn_model.model is not None:
                models_for_benchmark['lightweight_cnn'] = self.cnn_model.model
            
            X_test = self.processed_data['X_test']
            benchmark_results = self.evaluator.benchmark_real_time_performance(
                models_for_benchmark, X_test, batch_sizes=[1, 10, 100]
            )
        
        print(" Final report generated!")
        return report_path
    
    def _get_pipeline_summary(self):
        """Get summary of pipeline results"""
        summary = {
            'data_info': {
                'training_samples': self.processed_data['X_train'].shape[0],
                'validation_samples': self.processed_data['X_val'].shape[0],
                'test_samples': self.processed_data['X_test'].shape[0],
                'features': self.processed_data['X_train'].shape[1]
            },
            'models_trained': list(self.baseline_results.keys()) + (['lightweight_cnn'] if self.cnn_results else []),
            'best_model': None,
            'requirements_met': {}
        }
        
        if self.comparison_results is not None and not self.comparison_results.empty:
            # Find best model by F1-score
            best_idx = self.comparison_results['Val_F1_Score'].idxmax()
            best_model = self.comparison_results.iloc[best_idx]
            summary['best_model'] = {
                'name': best_model['Model'],
                'accuracy': best_model['Val_Accuracy'],
                'f1_score': best_model['Val_F1_Score'],
                'inference_time': best_model['Inference_Time_ms'],
                'memory_usage': best_model['Memory_Usage_MB']
            }
            
            # Check requirements compliance
            for _, row in self.comparison_results.iterrows():
                meets_accuracy = row['Val_Accuracy'] >= 0.90
                meets_speed = row['Inference_Time_ms'] <= 10.0
                meets_memory = row['Memory_Usage_MB'] <= 4000
                
                summary['requirements_met'][row['Model']] = {
                    'accuracy_90%': meets_accuracy,
                    'speed_10ms': meets_speed,
                    'memory_4gb': meets_memory,
                    'overall_pass': all([meets_accuracy, meets_speed, meets_memory])
                }
        
        return summary
    
    def print_final_summary(self, summary):
        """Print final pipeline summary"""
        print("\n" + "="*60)
        print(" PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        print(f"\n Dataset Information:")
        print(f"   Training Samples: {summary['data_info']['training_samples']:,}")
        print(f"   Validation Samples: {summary['data_info']['validation_samples']:,}")
        print(f"   Test Samples: {summary['data_info']['test_samples']:,}")
        print(f"   Features: {summary['data_info']['features']}")
        
        print(f"\n Models Trained: {len(summary['models_trained'])}")
        for model in summary['models_trained']:
            print(f"   ✓ {model.replace('_', ' ').title()}")
        
        if summary['best_model']:
            print(f"\n Best Performing Model:")
            best = summary['best_model']
            print(f"   Model: {best['name']}")
            print(f"   Accuracy: {best['accuracy']:.4f}")
            print(f"   F1-Score: {best['f1_score']:.4f}")
            print(f"   Inference Time: {best['inference_time']:.2f}ms")
            print(f"   Memory Usage: {best['memory_usage']:.2f}MB")
        
        print(f"\n Requirements Compliance (>90% Acc, ≤10ms, ≤4GB):")
        for model, compliance in summary['requirements_met'].items():
            status = "PASS " if compliance['overall_pass'] else "FAIL ❌"
            print(f"   {model}: {status}")
            print(f"      Accuracy: {'✓' if compliance['accuracy_90%'] else '✗'}")
            print(f"      Speed: {'✓' if compliance['speed_10ms'] else '✗'}")
            print(f"      Memory: {'✓' if compliance['memory_4gb'] else '✗'}")

def main():
    """
    Main execution function
    """
    print("LIGHTWEIGHT CNN FOR NETWORK INTRUSION DETECTION")
    print("Execution Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Research Question: Effective lightweight CNN architecture for binary intrusion detection")
    
    # Initialize and run pipeline
    pipeline = IntrustionDetectionPipeline()
    
    # Run complete pipeline
    # Use quick_run=True for faster execution in demo
    summary = pipeline.run_complete_pipeline(sample_size=8000, quick_run=True)
    
    # Print final summary
    pipeline.print_final_summary(summary)
    
    print(f"\n All results and models saved in:")
    print(f"   Data: {pipeline.data_dir}")
    print(f"   Models: {pipeline.models_dir}")
    print(f"   Results: {pipeline.results_dir}")
    
    print(f"\nProject completed successfully!")
    print(f" Check the results directory for detailed analysis and visualizations.")
    
    return pipeline, summary

if __name__ == "__main__":
    pipeline, summary = main()