"""
Lightweight CNN Implementation for Network Intrusion Detection
Focuses on efficiency while maintaining high accuracy
"""

import numpy as np
import pandas as pd
import time
import os
import psutil
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, provide sklearn fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    print(" TensorFlow available")
except ImportError:
    # Fallback to sklearn MLPClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
    TENSORFLOW_AVAILABLE = False
    print("  TensorFlow not available, using sklearn MLPClassifier fallback")

class LightweightCNN:
    def __init__(self, input_shape=None):
        """
        Initialize Lightweight CNN
        
        Args:
            input_shape: Shape of input features (for TensorFlow version)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.results = {}
        
    def create_cnn_model(self, depth=4, initial_filters=64, dropout_rate=0.3, 
                        learning_rate=0.001):
        """
        Create lightweight CNN architecture
        
        Args:
            depth: Number of convolutional layers (2, 4, or 6)
            initial_filters: Starting number of filters
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled CNN model
        """
        if not TENSORFLOW_AVAILABLE:
            print("  TensorFlow not available, cannot create CNN model")
            return None
            
        if self.input_shape is None:
            raise ValueError("Input shape must be specified for CNN model")
        
        print(f"  Creating {depth}-layer CNN with {initial_filters} initial filters...")
        
        model = keras.Sequential()
        
        # Reshape input for 1D CNN (treating features as sequence)
        model.add(layers.Reshape((self.input_shape[0], 1), input_shape=self.input_shape))
        
        # Progressive filter reduction: 64->32->16->8 (for depth=4)
        current_filters = initial_filters
        
        for i in range(depth):
            # 1D Convolutional layer
            model.add(layers.Conv1D(
                filters=current_filters,
                kernel_size=3,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            
            # Batch normalization for stable training
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            
            # Max pooling every 2 layers
            if i % 2 == 1:
                model.add(layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}'))
            
            # Dropout for regularization
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
            
            # Reduce filters progressively
            current_filters = max(8, current_filters // 2)
        
        # Global average pooling to reduce parameters
        model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout_rate, name='dropout_final'))
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        # Print model summary
        print(" Model Architecture:")
        model.summary()
        
        # Calculate model size
        param_count = model.count_params()
        print(f" Total Parameters: {param_count:,}")
        print(f" Estimated Model Size: {param_count * 4 / (1024**2):.2f} MB")
        
        return model
    
    def create_mlp_fallback(self, hidden_layers=(64, 32, 16), learning_rate=0.001):
        """
        Create MLP fallback when TensorFlow is not available
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            learning_rate: Learning rate for optimizer
            
        Returns:
            Configured MLPClassifier
        """
        print(f"  Creating MLP fallback with layers {hidden_layers}...")
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=512,
              depth=4, initial_filters=64, verbose=1):
        """
        Train the lightweight CNN or MLP
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            depth: CNN depth (ignored for MLP)
            initial_filters: Initial CNN filters (ignored for MLP)
            verbose: Verbosity level
            
        Returns:
            Training history and metrics
        """
        print(f" Starting training...")
        
        # Record memory usage before training
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        start_time = time.time()
        
        if TENSORFLOW_AVAILABLE:
            # TensorFlow CNN training
            if self.model is None:
                self.input_shape = (X_train.shape[1],)
                self.create_cnn_model(depth=depth, initial_filters=initial_filters)
            
            # Callbacks for better training
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.history = history.history
            
        else:
            # Sklearn MLP training
            if self.model is None:
                self.create_mlp_fallback()
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Create simple history for consistency
            self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        training_time = time.time() - start_time
        
        # Memory usage after training
        memory_after = process.memory_info().rss / (1024**2)  # MB
        memory_used = memory_after - memory_before
        
        # Validation predictions and metrics
        start_time = time.time()
        if TENSORFLOW_AVAILABLE:
            y_pred_val = (self.model.predict(X_val) > 0.5).astype(int).flatten()
            y_pred_proba_val = self.model.predict(X_val).flatten()
        else:
            y_pred_val = self.model.predict(X_val)
            y_pred_proba_val = self.model.predict_proba(X_val)[:, 1]
        
        inference_time = (time.time() - start_time) / len(X_val) * 1000  # ms per sample
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred_val, y_pred_proba_val)
        
        # Store results
        self.results = {
            'metrics': metrics,
            'training_time': training_time,
            'inference_time_ms': inference_time,
            'memory_usage_mb': memory_used,
            'model_params': {
                'depth': depth if TENSORFLOW_AVAILABLE else len(self.model.hidden_layer_sizes),
                'initial_filters': initial_filters if TENSORFLOW_AVAILABLE else 'N/A',
                'total_parameters': self.model.count_params() if TENSORFLOW_AVAILABLE else 'Unknown'
            }
        }
        
        print(f"\n Training completed!")
        print(f"     Training Time: {training_time:.2f}s")
        print(f"    Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Validation F1-Score: {metrics['f1_score']:.4f}")
        print(f"    Validation AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"    Inference Time: {inference_time:.2f}ms per sample")
        print(f"    Memory Usage: {memory_used:.2f}MB")
        
        return self.history, self.results
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Test metrics and predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        print(" Evaluating on test set...")
        
        # Predictions
        start_time = time.time()
        if TENSORFLOW_AVAILABLE:
            y_pred = (self.model.predict(X_test) > 0.5).astype(int).flatten()
            y_pred_proba = self.model.predict(X_test).flatten()
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        inference_time = (time.time() - start_time) / len(X_test) * 1000
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        test_results = {
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
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def benchmark_architectures(self, X_train, y_train, X_val, y_val, 
                              depths=[2, 4, 6], filter_sizes=[32, 64, 128]):
        """
        Benchmark different CNN architectures
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            depths: List of depths to test
            filter_sizes: List of initial filter sizes to test
            
        Returns:
            Comparison results
        """
        if not TENSORFLOW_AVAILABLE:
            print("  Architecture benchmarking requires TensorFlow")
            return None
        
        print(" Benchmarking CNN architectures...")
        
        benchmark_results = []
        
        for depth in depths:
            for initial_filters in filter_sizes:
                print(f"\n Testing depth={depth}, filters={initial_filters}")
                
                # Reset model
                self.model = None
                
                # Train with current configuration
                _, results = self.train(
                    X_train, y_train, X_val, y_val,
                    depth=depth, 
                    initial_filters=initial_filters,
                    epochs=20,  # Reduced for benchmarking
                    verbose=0
                )
                
                # Store results
                benchmark_results.append({
                    'depth': depth,
                    'initial_filters': initial_filters,
                    'accuracy': results['metrics']['accuracy'],
                    'f1_score': results['metrics']['f1_score'],
                    'auc_roc': results['metrics']['auc_roc'],
                    'training_time': results['training_time'],
                    'inference_time_ms': results['inference_time_ms'],
                    'memory_usage_mb': results['memory_usage_mb'],
                    'parameters': results['model_params']['total_parameters']
                })
        
        benchmark_df = pd.DataFrame(benchmark_results)
        
        print("\n Architecture Benchmark Results:")
        print(benchmark_df.round(4))
        
        # Find best configuration
        best_idx = benchmark_df['f1_score'].idxmax()
        best_config = benchmark_df.iloc[best_idx]
        
        print(f"\n Best Configuration:")
        print(f"   Depth: {best_config['depth']}")
        print(f"   Initial Filters: {best_config['initial_filters']}")
        print(f"   F1-Score: {best_config['f1_score']:.4f}")
        print(f"   Inference Time: {best_config['inference_time_ms']:.2f}ms")
        
        return benchmark_df
    
    def save_model(self, save_path):
        """Save trained model"""
        if self.model is None:
            print("  No model to save")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if TENSORFLOW_AVAILABLE:
            self.model.save(save_path)
        else:
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(self.model, f)
        
        print(f" Model saved to {save_path}")

def main():
    """
    Demonstration of lightweight CNN training
    """
    # Import data loader
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
    
    print(f"\n Dataset shapes:")
    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Initialize CNN
    cnn = LightweightCNN()
    
    print("\n" + "="*50)
    print(" TRAINING LIGHTWEIGHT CNN")
    print("="*50)
    
    # Train CNN
    history, results = cnn.train(
        X_train, y_train, X_val, y_val,
        epochs=30,
        depth=4,
        initial_filters=64
    )
    
    print("\n" + "="*50)
    print(" TESTING ON HELD-OUT DATA")
    print("="*50)
    
    # Evaluate on test set
    test_results = cnn.evaluate(X_test, y_test)
    
    # Architecture benchmarking (if TensorFlow available)
    if TENSORFLOW_AVAILABLE:
        print("\n" + "="*50)
        print(" ARCHITECTURE BENCHMARKING")
        print("="*50)
        
        benchmark_results = cnn.benchmark_architectures(
            X_train, y_train, X_val, y_val,
            depths=[2, 4], 
            filter_sizes=[32, 64]
        )
    
    # Save model
    model_path = "/home/user/intrusion_detection_project/models/lightweight_cnn"
    cnn.save_model(model_path)
    
    print("\n Lightweight CNN training and evaluation completed!")
    
    return cnn, test_results

if __name__ == "__main__":
    main()