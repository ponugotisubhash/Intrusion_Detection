"""
Baseline Machine Learning Models for Intrusion Detection
Implements Random Forest and XGBoost classifiers
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import time
import psutil
import os
from typing import Dict, Tuple, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, use sklearn alternative if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_AVAILABLE = False
    print("  XGBoost not available, using GradientBoostingClassifier instead")

class BaselineModels:
    def __init__(self):
        """Initialize baseline models"""
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def train_random_forest(self, X_train, y_train, X_val, y_val, 
                           n_estimators=100, max_depth=None, random_state=42):
        """
        Train Random Forest classifier
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            random_state: Random seed
            
        Returns:
            Trained model and performance metrics
        """
        print(f" Training Random Forest (n_estimators={n_estimators})...")
        
        # Record memory usage before training
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Initialize and train model
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Memory usage after training
        memory_after = process.memory_info().rss / (1024**2)  # MB
        memory_used = memory_after - memory_before
        
        # Predictions and evaluation
        start_time = time.time()
        y_pred_val = rf_model.predict(X_val)
        y_pred_proba_val = rf_model.predict_proba(X_val)[:, 1]
        inference_time = (time.time() - start_time) / len(X_val) * 1000  # ms per sample
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred_val, y_pred_proba_val)
        
        # Store results
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'metrics': metrics,
            'training_time': training_time,
            'inference_time_ms': inference_time,
            'memory_usage_mb': memory_used,
            'model_params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
        }
        
        # Feature importance
        self.feature_importance['random_forest'] = rf_model.feature_importances_
        
        print(f"    Training completed in {training_time:.2f}s")
        print(f"    Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Validation F1-Score: {metrics['f1_score']:.4f}")
        print(f"    Inference Time: {inference_time:.2f}ms per sample")
        print(f"   Memory Usage: {memory_used:.2f}MB")
        
        return rf_model, self.results['random_forest']
    
    def train_xgboost(self, X_train, y_train, X_val, y_val,
                     n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """
        Train XGBoost classifier (or GradientBoosting if XGBoost unavailable)
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            random_state: Random seed
            
        Returns:
            Trained model and performance metrics
        """
        model_name = 'xgboost' if XGBOOST_AVAILABLE else 'gradient_boosting'
        print(f" Training {model_name.replace('_', ' ').title()}...")
        
        # Record memory usage before training
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        if XGBOOST_AVAILABLE:
            # XGBoost implementation
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            # Scikit-learn GradientBoosting fallback
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
        
        start_time = time.time()
        
        if XGBOOST_AVAILABLE:
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
            
        training_time = time.time() - start_time
        
        # Memory usage after training
        memory_after = process.memory_info().rss / (1024**2)  # MB
        memory_used = memory_after - memory_before
        
        # Predictions and evaluation
        start_time = time.time()
        y_pred_val = model.predict(X_val)
        y_pred_proba_val = model.predict_proba(X_val)[:, 1]
        inference_time = (time.time() - start_time) / len(X_val) * 1000  # ms per sample
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred_val, y_pred_proba_val)
        
        # Store results
        self.models[model_name] = model
        self.results[model_name] = {
            'metrics': metrics,
            'training_time': training_time,
            'inference_time_ms': inference_time,
            'memory_usage_mb': memory_used,
            'model_params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate
            }
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = model.feature_importances_
        
        print(f"    Training completed in {training_time:.2f}s")
        print(f"    Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Validation F1-Score: {metrics['f1_score']:.4f}")
        print(f"    Inference Time: {inference_time:.2f}ms per sample")
        print(f"    Memory Usage: {memory_used:.2f}MB")
        
        return model, self.results[model_name]
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'auc_roc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def evaluate_on_test(self, X_test, y_test, model_names=None):
        """
        Evaluate trained models on test set
        
        Args:
            X_test, y_test: Test data
            model_names: List of model names to evaluate (None for all)
            
        Returns:
            Test results for all models
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        test_results = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                print(f"  Model {model_name} not found, skipping...")
                continue
                
            print(f" Evaluating {model_name.replace('_', ' ').title()} on test set...")
            
            model = self.models[model_name]
            
            # Predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            inference_time = (time.time() - start_time) / len(X_test) * 1000
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            test_results[model_name] = {
                'metrics': metrics,
                'inference_time_ms': inference_time,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"    Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Test F1-Score: {metrics['f1_score']:.4f}")
            print(f"    Test AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"    Test Inference Time: {inference_time:.2f}ms per sample")
        
        return test_results
    
    def compare_models(self, test_results=None):
        """
        Compare performance of all trained models
        
        Args:
            test_results: Test results (if available)
            
        Returns:
            Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Val_Accuracy': results['metrics']['accuracy'],
                'Val_F1_Score': results['metrics']['f1_score'],
                'Val_AUC_ROC': results['metrics']['auc_roc'],
                'Training_Time_s': results['training_time'],
                'Inference_Time_ms': results['inference_time_ms'],
                'Memory_Usage_MB': results['memory_usage_mb']
            }
            
            # Add test results if available
            if test_results and model_name in test_results:
                test_metrics = test_results[model_name]['metrics']
                row.update({
                    'Test_Accuracy': test_metrics['accuracy'],
                    'Test_F1_Score': test_metrics['f1_score'],
                    'Test_AUC_ROC': test_metrics['auc_roc']
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def get_feature_importance(self, model_name, feature_names, top_k=20):
        """
        Get top k most important features for a model
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.feature_importance:
            print(f"⚠️  Feature importance not available for {model_name}")
            return None
        
        importance_scores = self.feature_importance[model_name]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False).head(top_k)
        
        return feature_importance_df
    
    def save_models(self, save_dir):
        """Save trained models to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f" Saved {model_name} to {model_path}")
        
        # Save results
        results_path = os.path.join(save_dir, "baseline_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f" Saved results to {results_path}")

def main():
    """
    Demonstration of baseline models training
    """
    # This would typically import from data_loader
    from data_loader import CICIDS2017DataLoader
    
    print(" Loading and preprocessing data...")
    loader = CICIDS2017DataLoader()
    df = loader.generate_synthetic_data(n_samples=5000)  # Smaller for demo
    processed_data = loader.preprocess_data(df, apply_smote=True)
    
    # Extract data
    X_train = processed_data['X_train']
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_val = processed_data['y_val']
    y_test = processed_data['y_test']
    feature_names = processed_data['feature_names']
    
    print(f"\n Dataset shapes:")
    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    print("\n" + "="*50)
    print(" TRAINING BASELINE MODELS")
    print("="*50)
    
    # Train Random Forest
    rf_model, rf_results = baseline.train_random_forest(
        X_train, y_train, X_val, y_val, n_estimators=50  # Reduced for demo
    )
    
    print("\n" + "-"*50)
    
    # Train XGBoost (or GradientBoosting)
    xgb_model, xgb_results = baseline.train_xgboost(
        X_train, y_train, X_val, y_val, n_estimators=50  # Reduced for demo
    )
    
    print("\n" + "="*50)
    print(" TESTING ON HELD-OUT DATA")
    print("="*50)
    
    # Evaluate on test set
    test_results = baseline.evaluate_on_test(X_test, y_test)
    
    print("\n" + "="*50)
    print(" MODEL COMPARISON")
    print("="*50)
    
    # Compare models
    comparison_df = baseline.compare_models(test_results)
    print("\n Performance Comparison:")
    print(comparison_df.round(4))
    
    # Feature importance analysis
    print("\n Top 10 Most Important Features:")
    for model_name in baseline.models.keys():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        feature_imp = baseline.get_feature_importance(model_name, feature_names, top_k=10)
        if feature_imp is not None:
            for idx, row in feature_imp.iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save models
    print(f"\n Saving models...")
    baseline.save_models("/home/user/intrusion_detection_project/models")
    
    print("\n Baseline model training and evaluation completed!")
    
    return baseline, test_results, comparison_df

if __name__ == "__main__":
    main()