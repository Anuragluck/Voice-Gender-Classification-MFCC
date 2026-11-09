# EC24313 Assessment 2 Voice Gender Classification
**Birla Institute of Technology Mesra, Ranchi | Department of Electronics and Communication Engineering**  
**Student:** Anurag Tiwari | **Roll No.:** BTECH/10142/24  
**GitHub Repository:** [https://github.com/Anuragluck/Voice-Gender-Classification-MFCC](https://github.com/Anuragluck/Voice-Gender-Classification-MFCC)  

This project implements Assessment 2 using the class voice dataset. It ingests 148 audio files across Class 1 (Boys) and Class 2 (Girls), standardizes audio with a -30 dB silence trimming threshold (extending Assessment 1), extracts 13 Mel-Frequency Cepstral Coefficients (MFCCs), visualizes the acoustic space (PCA 2D scatter and direct MFCC 2 vs 3 scatter), benchmarks multiple classifiers (SVM, Random Forest, MLP Neural Network), and thoroughly evaluates performance using an out-of-fold Confusion Matrix, Accuracy, Sensitivity, Precision, and ROC-AUC.

## Project Deliverables & Layout

- `Anurag_Tiwari_Assessment2_Voice_Gender_Classification.ipynb` - Complete, self-contained, pre-executed Jupyter Notebook ready for submission and VS Code inspection.
- `src/run_assignment2.py` - Complete reproducible analysis and training pipeline.
- `src/build_notebook.py` - Builds and renders the full interactive Jupyter Notebook.
- `src/build_report.py` - Generates the comprehensive Word DOCX report.
- `src/build_pdf_report.py` - Generates the publication-quality submission-ready PDF report with full algorithm code.
- `outputs/` - Generated features, figures, model, benchmarks, metrics, and out-of-fold predictions.
- `report/` - Final DOCX and PDF submission report files:
  - `report/Anurag_Tiwari_Assessment2_Report.pdf`
  - `report/Anurag_Tiwari_Assessment2_Report.docx`

## Quick Run in VS Code

From the project root in VS Code terminal:

```powershell
.\.venv\Scripts\python.exe src\run_assignment2.py
.\.venv\Scripts\python.exe src\build_report.py
.\.venv\Scripts\python.exe src\build_pdf_report.py
```

You can also directly open `Anurag_Tiwari_Assessment2_Voice_Gender_Classification.ipynb` in VS Code to inspect the interactive cells, outputs, and embedded figures.

## Methodological Highlights

1. **Continuous Preprocessing with Assessment 1:** Resamples to 16 kHz mono, peak-normalizes to unity scale, and trims unvoiced/silence regions below -30 dB.
2. **13 MFCC Summary Vectors:** Computes 13 cepstral coefficients per frame and aggregates them via mean and standard deviation (26 numeric features).
3. **Zero Speaker Identity Leakage:** Employs 5-fold `StratifiedGroupKFold` on 76 inferred speaker groups. Paired `BIT` and `Name` recordings from the same individual strictly remain in the same fold.
4. **Class Balancing:** Overcomes severe 87.2% vs 12.8% imbalance via `class_weight='balanced'` in the RBF Support Vector Machine.
5. **Academic Transparency:** Highlights real-world dataset characteristics, including the dual-folder placement of the `Shreya` recordings, without silent manipulation of source data.

## Outputs Summary

- `outputs/dataset_manifest.csv` - Decoded metadata for all 148 recordings.
- `outputs/mfcc_features.csv` - The 26-dimensional feature table.
- `outputs/model_comparison.csv` - Benchmark comparison of SVM (RBF), SVM (Linear), Random Forest, Logistic Regression, and MLP Neural Network.
- `outputs/01_class_distribution.png` - Class imbalance bar chart.
- `outputs/02_mfcc_pca_scatter.png` - 2D PCA projection scatter plot.
- `outputs/02b_mfcc_scatter_direct.png` - Direct acoustic MFCC 2 vs MFCC 3 scatter plot.
- `outputs/03_confusion_matrix.png` - Out-of-fold confusion matrix.
- `outputs/04_roc_curve.png` - ROC curve and AUC (0.910).
- `outputs/metrics.json` and `outputs/run_summary.txt` - Numerical metrics.
- `outputs/oof_predictions.csv` - Out-of-fold predictions for every individual utterance.
