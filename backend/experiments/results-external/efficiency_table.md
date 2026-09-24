| Model | Model Size (KB) | Training Time (10k rows) | Inference Latency (per URL) | Total Latency (Extraction + Model) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 2.09 KB | 0.029 s | 0.0001 ms | 0.0134 ms |
| Decision Tree | 3.62 KB | 0.011 s | 0.0001 ms | 0.0135 ms |
| Random Forest | 3135.49 KB | 0.394 s | 0.0041 ms | 0.0175 ms |
| SVM (RBF approximation) | 27.26 KB | 0.053 s | 0.0016 ms | 0.0149 ms |
| XGBoost | 235.16 KB | 0.448 s | 0.0007 ms | 0.0141 ms |