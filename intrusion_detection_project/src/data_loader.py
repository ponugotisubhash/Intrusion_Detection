"""
Data Loader and Preprocessor for CICIDS2017 Dataset
Handles loading, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class CICIDS2017DataLoader:
    def __init__(self, data_path=None, sample_size=None):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to CICIDS2017 CSV files
            sample_size: Number of samples to use (for testing)
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic network flow data for demonstration
        Based on CICIDS2017 feature structure
        """
        np.random.seed(42)
        
        # Generate realistic network flow features
        data = {
            'flow_duration': np.random.exponential(1000, n_samples),
            'total_fwd_packets': np.random.poisson(10, n_samples),
            'total_backward_packets': np.random.poisson(8, n_samples),
            'total_length_fwd_packets': np.random.exponential(1500, n_samples),
            'total_length_bwd_packets': np.random.exponential(1200, n_samples),
            'fwd_packet_length_max': np.random.gamma(2, 200, n_samples),
            'fwd_packet_length_min': np.random.gamma(1, 50, n_samples),
            'fwd_packet_length_mean': np.random.normal(200, 50, n_samples),
            'fwd_packet_length_std': np.random.gamma(1, 30, n_samples),
            'bwd_packet_length_max': np.random.gamma(2, 180, n_samples),
            'bwd_packet_length_min': np.random.gamma(1, 40, n_samples),
            'bwd_packet_length_mean': np.random.normal(180, 45, n_samples),
            'bwd_packet_length_std': np.random.gamma(1, 25, n_samples),
            'flow_bytes_s': np.random.exponential(10000, n_samples),
            'flow_packets_s': np.random.exponential(50, n_samples),
            'flow_iat_mean': np.random.exponential(100, n_samples),
            'flow_iat_std': np.random.exponential(200, n_samples),
            'flow_iat_max': np.random.exponential(500, n_samples),
            'flow_iat_min': np.random.exponential(10, n_samples),
            'fwd_iat_total': np.random.exponential(800, n_samples),
            'fwd_iat_mean': np.random.exponential(80, n_samples),
            'fwd_iat_std': np.random.exponential(150, n_samples),
            'fwd_iat_max': np.random.exponential(400, n_samples),
            'fwd_iat_min': np.random.exponential(8, n_samples),
            'bwd_iat_total': np.random.exponential(600, n_samples),
            'bwd_iat_mean': np.random.exponential(60, n_samples),
            'bwd_iat_std': np.random.exponential(120, n_samples),
            'bwd_iat_max': np.random.exponential(300, n_samples),
            'bwd_iat_min': np.random.exponential(6, n_samples),
            'fwd_psh_flags': np.random.binomial(1, 0.1, n_samples),
            'bwd_psh_flags': np.random.binomial(1, 0.08, n_samples),
            'fwd_urg_flags': np.random.binomial(1, 0.01, n_samples),
            'bwd_urg_flags': np.random.binomial(1, 0.008, n_samples),
            'fwd_header_length': np.random.normal(20, 5, n_samples),
            'bwd_header_length': np.random.normal(20, 5, n_samples),
            'fwd_packets_s': np.random.exponential(25, n_samples),
            'bwd_packets_s': np.random.exponential(20, n_samples),
            'min_packet_length': np.random.gamma(1, 20, n_samples),
            'max_packet_length': np.random.gamma(3, 300, n_samples),
            'packet_length_mean': np.random.normal(190, 40, n_samples),
            'packet_length_std': np.random.gamma(1, 35, n_samples),
            'packet_length_variance': np.random.gamma(2, 500, n_samples),
            'fin_flag_count': np.random.binomial(2, 0.05, n_samples),
            'syn_flag_count': np.random.binomial(2, 0.04, n_samples),
            'rst_flag_count': np.random.binomial(2, 0.02, n_samples),
            'psh_flag_count': np.random.binomial(3, 0.08, n_samples),
            'ack_flag_count': np.random.binomial(5, 0.6, n_samples),
            'urg_flag_count': np.random.binomial(1, 0.01, n_samples),
            'cwe_flag_count': np.random.binomial(1, 0.005, n_samples),
            'ece_flag_count': np.random.binomial(1, 0.003, n_samples),
            'down_up_ratio': np.random.gamma(2, 0.5, n_samples),
            'average_packet_size': np.random.normal(200, 50, n_samples),
            'avg_fwd_segment_size': np.random.normal(180, 45, n_samples),
            'avg_bwd_segment_size': np.random.normal(160, 40, n_samples),
            'fwd_avg_bytes_bulk': np.random.exponential(100, n_samples),
            'fwd_avg_packets_bulk': np.random.exponential(5, n_samples),
            'fwd_avg_bulk_rate': np.random.exponential(50, n_samples),
            'bwd_avg_bytes_bulk': np.random.exponential(80, n_samples),
            'bwd_avg_packets_bulk': np.random.exponential(4, n_samples),
            'bwd_avg_bulk_rate': np.random.exponential(40, n_samples),
            'subflow_fwd_packets': np.random.poisson(8, n_samples),
            'subflow_fwd_bytes': np.random.exponential(1200, n_samples),
            'subflow_bwd_packets': np.random.poisson(6, n_samples),
            'subflow_bwd_bytes': np.random.exponential(1000, n_samples),
            'init_win_bytes_forward': np.random.exponential(8000, n_samples),
            'init_win_bytes_backward': np.random.exponential(7000, n_samples),
            'act_data_pkt_fwd': np.random.poisson(7, n_samples),
            'min_seg_size_forward': np.random.gamma(1, 15, n_samples),
            'active_mean': np.random.exponential(200, n_samples),
            'active_std': np.random.exponential(300, n_samples),
            'active_max': np.random.exponential(800, n_samples),
            'active_min': np.random.exponential(50, n_samples),
            'idle_mean': np.random.exponential(1000, n_samples),
            'idle_std': np.random.exponential(2000, n_samples),
            'idle_max': np.random.exponential(5000, n_samples),
            'idle_min': np.random.exponential(100, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic attack patterns
        # 85% benign, 15% attacks (various types)
        labels = np.random.choice(['BENIGN', 'DoS', 'Brute Force', 'Web Attack', 'Infiltration'], 
                                 n_samples, p=[0.85, 0.06, 0.04, 0.03, 0.02])
        
        # Modify features for attack patterns
        attack_mask = labels != 'BENIGN'
        
        # DoS attacks - high packet rates, long durations
        dos_mask = labels == 'DoS'
        df.loc[dos_mask, 'flow_packets_s'] *= np.random.uniform(10, 50, dos_mask.sum())
        df.loc[dos_mask, 'flow_duration'] *= np.random.uniform(5, 20, dos_mask.sum())
        
        # Brute Force - many small packets, specific port patterns
        bf_mask = labels == 'Brute Force'
        df.loc[bf_mask, 'total_fwd_packets'] *= np.random.uniform(5, 15, bf_mask.sum())
        df.loc[bf_mask, 'fwd_packet_length_mean'] *= np.random.uniform(0.3, 0.7, bf_mask.sum())
        
        # Web attacks - specific HTTP patterns
        web_mask = labels == 'Web Attack'  
        df.loc[web_mask, 'average_packet_size'] *= np.random.uniform(1.5, 3, web_mask.sum())
        df.loc[web_mask, 'psh_flag_count'] *= np.random.uniform(2, 5, web_mask.sum())
        
        df['label'] = labels
        
        return df
    
    def load_real_data(self):
        """
        Load real CICIDS2017 data from CSV files
        This would be implemented when actual data files are available
        """
        if self.data_path is None:
            raise ValueError("Data path not provided for real data loading")
        
        # Implementation for loading real CICIDS2017 CSV files
        # This would iterate through the daily CSV files and combine them
        pass
    
    def preprocess_data(self, df, test_size=0.2, val_size=0.1, apply_smote=True):
        """
        Preprocess the dataset for machine learning
        
        Args:
            df: Input dataframe
            test_size: Proportion of test set
            val_size: Proportion of validation set from training data
            apply_smote: Whether to apply SMOTE for handling class imbalance
            
        Returns:
            Preprocessed train, validation, and test sets
        """
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Handle infinite and NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Encode labels for binary classification (BENIGN vs ATTACK)
        y_binary = (y != 'BENIGN').astype(int)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split into train and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE if requested (guarded against environment/library issues)
        if apply_smote:
            try:
                smote = SMOTE(random_state=42)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, np.asarray(y_train).ravel())
            except Exception as e:
                warnings.warn(f"SMOTE failed ({e}); continuing without resampling.")
                # Ensure y_train remains a numpy array for downstream code
                y_train = np.asarray(y_train).ravel()
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_names
        }
    
    def get_data_info(self, df):
        """
        Get comprehensive information about the dataset
        """
        info = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / (1024**2),  # MB
            'label_distribution': df['label'].value_counts(),
            'label_percentages': df['label'].value_counts(normalize=True) * 100,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        return info

def main():
    """
    Demonstration of data loading and preprocessing
    """
    print(" Initializing CICIDS2017 Data Loader...")
    loader = CICIDS2017DataLoader()
    
    print(" Generating synthetic dataset...")
    df = loader.generate_synthetic_data(n_samples=10000)
    
    print(" Dataset Information:")
    info = loader.get_data_info(df)
    print(f"   Shape: {info['shape']}")
    print(f"   Memory Usage: {info['memory_usage']:.2f} MB")
    print(f"   Missing Values: {info['missing_values']}")
    print(f"   Duplicate Rows: {info['duplicate_rows']}")
    print("\n Label Distribution:")
    for label, count in info['label_distribution'].items():
        percentage = info['label_percentages'][label]
        print(f"   {label}: {count} ({percentage:.2f}%)")
    
    print("\n Preprocessing data...")
    processed_data = loader.preprocess_data(df, apply_smote=True)
    
    print(" Preprocessing complete!")
    print(f"   Training set: {processed_data['X_train'].shape}")
    print(f"   Validation set: {processed_data['X_val'].shape}")
    print(f"   Test set: {processed_data['X_test'].shape}")
    print(f"   Features: {len(processed_data['feature_names'])}")
    
    return processed_data

if __name__ == "__main__":
    main()