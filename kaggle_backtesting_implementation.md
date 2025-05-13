# Kaggle Notebooks Backtesting Implementation

## Setup Guide

This guide details how to implement and optimize our trading algorithm backtesting on Kaggle Notebooks, leveraging datasets already stored on the platform.

## Initial Setup

### 1. Creating a Base Notebook

1. **Log in to Kaggle** and navigate to "Code" → "Create New Notebook"
2. **Configure notebook settings:**
   - Accelerator: GPU P100
   - Language: Python
   - Save notebook with descriptive name (e.g., `binance_futures_backtesting_v1`)

### 2. Environment Configuration

Add the following setup cell at the beginning of your notebook:

```python
# Install required packages
!pip install -q ccxt==4.4.80 pandas==2.2.3 numpy==2.2.5 matplotlib==3.10.3 pydantic==2.11.4 python-dotenv==1.1.0 PyYAML==6.0.2 psutil==7.0.0

# Verify CUDA is available (for GPU acceleration)
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 3. Repository Access

Choose one of these approaches:

**Option A: Clone project repository** (if public):
```python
# Clone the repository
!git clone https://github.com/yourusername/nancemakina.git
%cd nancemakina

# Add project root to Python path
import os
import sys
sys.path.append(os.getcwd())
```

**Option B: Upload key scripts** (if private):
```python
# Upload key script files from project (run this once)
# After upload, files appear in "../input/crypto-trading-bot/" directory
!mkdir -p scripts
!mkdir -p config
!mkdir -p src

# Copy uploaded files to working directory
!cp -r ../input/crypto-trading-bot/scripts/* ./scripts/
!cp -r ../input/crypto-trading-bot/config/* ./config/
!cp -r ../input/crypto-trading-bot/src/* ./src/
```

## Accessing Kaggle Datasets

### 1. Using Existing Dataset

If your dataset is already on Kaggle:

```python
# Load dataset (replace with your actual dataset path)
kaggle_dataset_path = "../input/binance-futures-data/merged_futures_data_quarter.csv"

# Verify dataset exists and show basic info
import os
print(f"Dataset exists: {os.path.exists(kaggle_dataset_path)}")
print(f"Dataset size: {os.path.getsize(kaggle_dataset_path) / (1024 * 1024):.2f} MB")

# Create symbolic link to expected location
!mkdir -p backtest_data
!ln -s {kaggle_dataset_path} backtest_data/merged_futures_data_quarter.csv
```

### 2. Creating Dataset (if needed)

If you need to create a new Kaggle dataset:

1. Go to "Data" → "New Dataset"
2. Upload your backtesting data files (within 20GB total limit)
3. Provide metadata and publish
4. Reference in your notebook using path: `../input/your-dataset-name/your-file.csv`

## Optimizing for Kaggle Constraints

### 1. Session Persistence Strategy

To handle the 20-minute idle timeout and 9-hour session limit:

```python
# Add this to functions that run long operations
def keep_session_alive():
    """Output something every 10 minutes to prevent timeout."""
    import threading
    import time
    
    def _print_status():
        while True:
            print(".", end="", flush=True)
            time.sleep(600)  # 10 minutes
    
    thread = threading.Thread(target=_print_status)
    thread.daemon = True
    thread.start()
```

### 2. Checkpoint System

Create checkpoints to resume work between sessions:

```python
def save_checkpoint(state, filename="backtest_checkpoint.pkl"):
    """Save backtest state for later resumption."""
    import pickle
    
    # Save to Kaggle's temporary storage
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    
    # Also save to output for download (Kaggle saves outputs between sessions)
    !mkdir -p /kaggle/working/outputs
    !cp {filename} /kaggle/working/outputs/
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename="backtest_checkpoint.pkl"):
    """Load backtest state from checkpoint."""
    import pickle
    import os
    
    # Try local file first
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    # Try from output directory
    output_path = f"/kaggle/working/outputs/{filename}"
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            return pickle.load(f)
            
    return None
```

### 3. Memory Optimization

Kaggle has 13GB RAM limit. Optimize memory usage with:

```python
def optimize_memory():
    """Release memory to prevent OOM errors."""
    import gc
    import psutil
    
    # Force garbage collection
    gc.collect()
    
    # Report memory usage
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

# Call this periodically during backtest
```

## Implementing the Backtest

### 1. Modifying Backtest Runner

Create a Kaggle-optimized entry point:

```python
def run_kaggle_backtest(config_path, data_path, symbol, config_symbol,
                        sample=1.0, chunk_mode="time", chunk_size=7, 
                        resume_from_checkpoint=True):
    """Run backtest optimized for Kaggle environment."""
    # Check for checkpoint and resume if available
    checkpoint = None
    if resume_from_checkpoint:
        checkpoint = load_checkpoint()
        if checkpoint:
            print(f"Resuming from checkpoint at iteration {checkpoint['iteration']}")
    
    # Keep session alive
    keep_session_alive()
    
    # Import core backtest functionality
    from scripts.backtest_optimized import run_backtest
    
    # Configure chunking to fit within Kaggle memory constraints
    chunk_opts = f"--chunk-mode {chunk_mode} --chunk-size {chunk_size}" 
    
    # Run backtest
    results = run_backtest(
        config=config_path,
        data=data_path,
        symbol=symbol,
        config_symbol=config_symbol,
        sample=sample,
        cache_opts="--no-cache",  # Kaggle storage is temporary
        chunk_opts=chunk_opts,
        checkpoint=checkpoint
    )
    
    # Save checkpoint periodically within run_backtest
    # (requires modifying the original script)
    
    # Save final results
    results.save_to_kaggle()
    
    return results
```

### 2. Main Execution Cell

Create the main execution cell for your notebook:

```python
# MAIN EXECUTION CELL
# Run this cell to execute the backtest

# Configuration
CONFIG_PATH = "config/v1_optimized_strategy.yaml"
DATA_PATH = "backtest_data/merged_futures_data_quarter.csv"
SYMBOL = "BTCUSDT"
CONFIG_SYMBOL = "BTC_USDT"
SAMPLE_SIZE = 0.25  # Use 25% of data for quick testing
CHUNK_MODE = "time"
CHUNK_SIZE = 7  # days
RESUME = True

# Execute backtest
results = run_kaggle_backtest(
    config_path=CONFIG_PATH,
    data_path=DATA_PATH, 
    symbol=SYMBOL,
    config_symbol=CONFIG_SYMBOL,
    sample=SAMPLE_SIZE,
    chunk_mode=CHUNK_MODE,
    chunk_size=CHUNK_SIZE,
    resume_from_checkpoint=RESUME
)

# Display results
display(results.summary())
results.plot_equity_curve()

# Save results to output (accessible after notebook stops)
results.save_to_file("/kaggle/working/outputs/backtest_results.pkl")
```

## Visualization and Analysis

### 1. Results Visualization Cell

```python
# VISUALIZATION CELL
# Run this cell to visualize results

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
try:
    with open("/kaggle/working/outputs/backtest_results.pkl", "rb") as f:
        results = pickle.load(f)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results['equity_curve'])
    plt.title(f"Equity Curve - {SYMBOL}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.show()
    
    # Show trade statistics
    stats_df = pd.DataFrame(results['stats'])
    display(stats_df)
    
    # Plot drawdown
    plt.figure(figsize=(12, 6))
    plt.plot(results['drawdown'])
    plt.title(f"Drawdown - {SYMBOL}")
    plt.xlabel("Date")
    plt.ylabel("Drawdown %")
    plt.grid(True)
    plt.show()
    
except Exception as e:
    print(f"Error visualizing results: {e}")
```

### 2. Parameter Sweeping

Create a parameter sweep to test multiple configurations:

```python
# PARAMETER SWEEP CELL
# Test multiple parameter combinations

# Define parameter ranges to test
param_grid = {
    "stop_loss": [0.01, 0.02, 0.03],
    "take_profit": [0.02, 0.03, 0.04],
    "sma_short": [20, 21, 22],
    "sma_long": [195, 200, 205]
}

# Import itertools for parameter combinations
import itertools
import yaml
import copy
import time

# Load base config
with open(CONFIG_PATH, "r") as f:
    base_config = yaml.safe_load(f)

# Generate all combinations
keys = param_grid.keys()
values = param_grid.values()
combinations = list(itertools.product(*values))

# Run backtest for each combination
results_dict = {}
for i, combination in enumerate(combinations):
    # Create parameter set
    params = dict(zip(keys, combination))
    print(f"Testing combination {i+1}/{len(combinations)}: {params}")
    
    # Create modified config
    config = copy.deepcopy(base_config)
    for k, v in params.items():
        # Set nested keys correctly (adjust paths as needed)
        if k == "stop_loss" or k == "take_profit":
            config["risk_management"][k] = v
        elif k == "sma_short" or k == "sma_long":
            config["indicators"][k]["period"] = v
    
    # Save temporary config
    temp_config_path = f"config/temp_config_{i}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)
    
    # Run backtest with this config
    try:
        result = run_kaggle_backtest(
            config_path=temp_config_path,
            data_path=DATA_PATH,
            symbol=SYMBOL,
            config_symbol=CONFIG_SYMBOL,
            sample=SAMPLE_SIZE,
            resume_from_checkpoint=False  # Fresh run for each config
        )
        
        # Store results
        key = "_".join([f"{k}={v}" for k, v in params.items()])
        results_dict[key] = result
        
        # Save intermediate results
        with open("/kaggle/working/outputs/param_sweep_results.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # Cleanup and optimize memory
        optimize_memory()
        
    except Exception as e:
        print(f"Error with combination {i}: {e}")
    
    # Prevent timeouts
    time.sleep(1)  # Brief pause between runs

# Display summary of all results
summary_rows = []
for key, result in results_dict.items():
    summary = result.summary()
    summary["parameters"] = key
    summary_rows.append(summary)

summary_df = pd.DataFrame(summary_rows)
display(summary_df.sort_values(by="profit", ascending=False))
```

## Semi-Automation Approach

### 1. Scheduled Re-execution

Since Kaggle doesn't support true automation, use these approaches to semi-automate:

1. **Set calendar reminders** to manually restart notebook at regular intervals
2. **Create status cell** that shows progress and next steps:

```python
# STATUS CELL - Run this to check progress
import os
import glob
import time

# Check progress files
checkpoint_files = glob.glob("/kaggle/working/outputs/*.pkl")
print(f"Found {len(checkpoint_files)} checkpoint/result files:")
for file in checkpoint_files:
    print(f" - {file} ({time.ctime(os.path.getmtime(file))})")

# Check memory usage
optimize_memory()

# Check GPU status
!nvidia-smi

# Display next steps
if os.path.exists("/kaggle/working/outputs/backtest_results.pkl"):
    print("✅ Main backtest complete - run visualization cell")
else:
    print("⏳ Main backtest not complete - run or resume main execution cell")
    
if os.path.exists("/kaggle/working/outputs/param_sweep_results.pkl"):
    print("✅ Parameter sweep in progress/complete - check results")
else:
    print("❌ Parameter sweep not started - run parameter sweep cell")
```

### 2. Email Notification (Optional)

If manually checking is too burdensome, set up email notifications:

```python
# Requires kaggle API token configured
def send_notification(subject, message):
    """Send notification when task completes or needs attention."""
    import smtplib
    from email.mime.text import MIMEText
    import os
    
    # For security, use Kaggle secrets for credentials
    # (Configure in Kaggle account settings)
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        email = user_secrets.get_secret("notification_email")
        password = user_secrets.get_secret("email_password")
        recipient = user_secrets.get_secret("recipient_email")
        
        # Create message
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = email
        msg['To'] = recipient
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email, password)
        server.send_message(msg)
        server.quit()
        
        print(f"Notification sent: {subject}")
        
    except Exception as e:
        print(f"Failed to send notification: {e}")
```

## Best Practices for Kaggle

1. **Break large backtests into smaller chunks**:
   - Use smaller timeframes initially
   - Validate on small samples before full runs

2. **Keep notebook cells modular**:
   - One cell for setup
   - One cell for data loading
   - One cell for execution
   - One cell for visualization

3. **Monitor resource usage**:
   - Add periodic memory checks
   - Watch for GPU utilization

4. **Handle disconnections gracefully**:
   - Save state frequently
   - Design for resumption
   - Keep checkpoints in output directory

5. **Optimize dataset usage**:
   - Use Kaggle datasets feature for larger data
   - Consider preprocessing data before upload

## Example Workflow

1. **Monday**: Upload data to Kaggle dataset
2. **Tuesday**: Create and configure notebook 
3. **Wednesday**: Run initial tests with sample data
4. **Thursday**: Start full backtest, monitor and restart as needed
5. **Friday**: Analyze results and perform parameter sweeps
6. **Weekend**: Download final results and analyze locally

## Typical Commands for Terminal in Notebook

Kaggle notebooks include a terminal that can be useful for debugging:

```bash
# List available files
!ls -la

# Check memory usage
!free -h

# Check GPU status
!nvidia-smi

# Check dataset size
!du -sh /kaggle/input/*

# Kill hanging processes
!pkill -9 python
```

## Conclusion

By following this implementation guide, you can effectively leverage Kaggle Notebooks for your trading algorithm backtesting. While not fully automated, this approach provides a robust framework for running extensive backtests with minimal manual intervention, taking advantage of Kaggle's free GPU resources and your existing datasets. 