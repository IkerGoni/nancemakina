# Automated Cloud Backtesting Development Plan

## Overview

This development plan outlines the strategy for automating backtesting of our trading algorithms using free cloud computing resources. The goal is to leverage external computing power for extensive testing while maintaining consistency with our local development environment.

## Cloud Platform Options

### 1. Kaggle Notebooks

**Resources:**
- Tesla P100 GPU (13GB VRAM)
- 2 CPU cores
- 13GB RAM
- 20GB storage across all notebooks

**Limitations:**
- 30 hours/week GPU time limit
- 9-hour maximum execution per session
- 20-minute idle timeout
- No true automation (requires manual interaction)

### 2. Google Colab

**Resources:**
- T4 or P100 GPU
- ~12GB RAM
- 12-hour runtime limit
- File storage through Google Drive

**Limitations:**
- Disconnects after 12 hours
- Variable idle timeout (30-90 minutes)
- Limited persistent storage

### 3. GitHub Actions

**Resources:**
- Free tier: 2,000 minutes/month
- 2-core CPU (no GPU)
- 7GB RAM
- Can run on schedule

**Limitations:**
- No GPU access
- 6-hour maximum runtime
- 5GB artifact storage

## Implementation Plan

### Phase 1: Cloud Environment Setup (Week 1)

1. **Create adaptable script wrappers:**
   - Modify `run_optimized_backtest.sh` to work in notebook environments
   - Create Python entry point that can run without shell access

2. **Package dependencies management:**
   - Create condensed `requirements-cloud.txt` with essential packages
   - Ensure compatibility with cloud notebook environments

3. **Data management solution:**
   - Implement data fetcher scripts to download minimal required data
   - Set up cloud storage synchronization (Google Drive/Kaggle Datasets)

### Phase 2: Notebook Adaptation (Week 2)

1. **Create templated notebook files:**
   - Kaggle notebook template with installation and execution cells
   - Google Colab template with equivalent functionality
   - Add result visualization and export components

2. **Implement parameterization:**
   - Add configuration cells for easy parameter adjustment
   - Create parameter sweep capability for testing multiple configurations

3. **Result persistence:**
   - Set up automatic result export to cloud storage
   - Implement checkpoint saving to resume interrupted runs

### Phase 3: Automation Framework (Week 3)

1. **GitHub Actions workflow:**
   - Create workflow for non-GPU preprocessing and analysis
   - Set up scheduled runs for regular testing
   - Implement result aggregation and notification system

2. **Semi-automated notebook execution:**
   - Script to generate notebooks with different parameters
   - Create dashboard for tracking running experiments

3. **Result analysis automation:**
   - Develop comparison scripts for evaluating strategy variations
   - Create visualization notebook for performance metrics

## Technical Solutions

### Running Backtests in Notebooks

```python
# Example notebook cell structure
!pip install -q ccxt pandas numpy matplotlib pydantic python-dotenv

# Clone repo (or upload key files)
!git clone https://github.com/yourusername/nancemakina.git
%cd nancemakina

# Set up environment
import os
import sys
sys.path.append(os.getcwd())

# Run backtest with parameters
from scripts.backtest_optimized import run_backtest

# Configure parameters
config_path = "config/v1_optimized_strategy.yaml"
data_path = "backtest_data/merged_futures_data_quarter.csv"
symbol = "BTCUSDT"
config_symbol = "BTC_USDT"
sample_size = 0.25  # Use 25% of data for quick testing

# Execute backtest
results = run_backtest(
    config=config_path,
    data=data_path,
    symbol=symbol,
    config_symbol=config_symbol,
    sample=sample_size,
    cache_opts="--no-cache",  # Since cloud storage is temporary
    chunk_opts="--chunk-mode time --chunk-size 7"
)

# Visualize and export results
display(results.summary())
results.plot_equity_curve()
results.export_to_drive("backtest_results_btc.pkl")
```

### Data Management Solutions

1. **For larger datasets:**
   - Split data into manageable chunks
   - Store in cloud storage (Google Drive, Kaggle Datasets)
   - Download only needed segments when running tests

2. **Preprocessing optimization:**
   - Run heavy preprocessing locally once
   - Upload preprocessed data to cloud storage
   - Use the preprocessed data in cloud environments

### Testing Automation Options

1. **Scheduled GitHub Actions:**
   ```yaml
   # .github/workflows/weekly-backtest.yml
   name: Weekly Backtest
   on:
     schedule:
       - cron: '0 0 * * 0'  # Run every Sunday
   jobs:
     preprocess:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.11'
         - name: Install dependencies
           run: pip install -r requirements-cloud.txt
         - name: Run preprocessing
           run: python scripts/prepare_backtest_data.py
         - name: Upload processed data
           uses: actions/upload-artifact@v3
           with:
             name: processed-data
             path: backtest_data/processed/
   ```

2. **Notebook generation script:**
   - Create script to generate notebooks with different parameters
   - Upload to Kaggle/Colab programmatically (using APIs)
   - Track running notebooks and collect results

## Limitations and Considerations

1. **Cloud execution limitations:**
   - Cannot run truly unattended due to timeouts
   - Need to handle disconnections and restarts
   - Limited by weekly free GPU quotas

2. **Data transfer bottlenecks:**
   - Cloud upload/download speeds can be slow
   - Need to optimize what data is transferred
   - Consider using sampling strategies more aggressively

3. **Environment consistency:**
   - Cloud environments may have different package versions
   - Need to ensure reproducibility across environments
   - Document specific environment configurations

4. **Security considerations:**
   - No API keys or sensitive data in notebooks
   - Use environment variables or secure storage options
   - Consider anonymous GitHub repository for public cloud testing

## Future Enhancements

1. **Hybrid cloud-local approach:**
   - Use cloud for parameter sweeps and initial testing
   - Use local environment for final validation
   - Develop synchronization mechanism between environments

2. **Custom modular testing framework:**
   - Extract core backtesting logic from visualization/UI
   - Create standardized result format for cross-platform comparison
   - Develop headless testing mode for CI/CD pipelines

3. **Commercial cloud options (if budget available):**
   - AWS SageMaker notebooks (~$0.10-$0.20/hour for basic instances)
   - Google Cloud Vertex AI (~$0.10/hour + storage)
   - Paperspace Gradient ($8/month for basic tier)

## Success Metrics

- Ability to run at least 5x more backtest configurations weekly
- Reduced local computational resource usage by 80%
- Test results within 2% variance across environments
- Automated performance reports for strategy improvements

## References

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions) 