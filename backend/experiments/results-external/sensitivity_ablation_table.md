| Feature Configuration | Count | In-Dataset F1 (PhiUSIIL CV) | External F1 (Wangchuk) | Inference Latency (ms/URL) |
| :--- | :---: | :---: | :---: | :---: |
| Full Deployed (12 features) | 12 | 99.49% | 3.81% | 0.00036 ms |
| Top-7 Features (MI-based) | 7 | 99.29% | 4.74% | 0.00031 ms |
| Top-5 Features (MI-based) | 5 | 99.27% | 4.77% | 0.00032 ms |
| Remove HTTPS Shortcut (11 features) | 11 | 99.24% | 3.77% | 0.00034 ms |
| Remove IP Address Indicator (11 features) | 11 | 99.49% | 3.81% | 0.00035 ms |