"""Build the comprehensive final DOCX report for EC24313 Assessment 2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_PATH = REPORT_DIR / "Anurag_Tiwari_Assessment2_Report.docx"
CODE_FILE = PROJECT_ROOT / "src" / "run_assignment2.py"

BLUE = "1F4E79"
LIGHT_BLUE = "DCE6F1"
LIGHT_GRAY = "F4F5F7"
BORDER_GRAY = "D9D9D9"


def set_font(run, name: str = "Aptos", size: float | None = None, bold: bool | None = None, italic: bool | None = None) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_border(cell, color: str = BORDER_GRAY) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), color)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run("Page ")
    set_font(run, size=9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)


def write_cell(cell, text: str, *, bold: bool = False, color: str | None = None, align=None, font_size: float = 9.5) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.space_before = Pt(0)
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(str(text))
    set_font(run, size=font_size, bold=bold)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_cell_border(cell)


def add_table(document: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = False
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.width = Inches(widths[index])
        set_cell_shading(cell, BLUE)
        write_cell(cell, header, bold=True, color="FFFFFF", align=WD_ALIGN_PARAGRAPH.CENTER, font_size=9.5)
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].width = Inches(widths[index])
            if row_index % 2 == 1:
                set_cell_shading(cells[index], LIGHT_GRAY)
            alignment = WD_ALIGN_PARAGRAPH.CENTER if index > 0 else WD_ALIGN_PARAGRAPH.LEFT
            write_cell(cells[index], value, align=alignment, font_size=9.0)
    document.add_paragraph().paragraph_format.space_after = Pt(4)


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(10)
    run = paragraph.add_run(text)
    set_font(run, size=9, italic=True)


def add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_heading(text, level=level)
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    for run in paragraph.runs:
        set_font(run, size=13.5 if level == 1 else 11.5, bold=True)
        run.font.color.rgb = RGBColor(0, 51, 102) if level == 1 else RGBColor(50, 50, 50)


def add_body(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.15
    run = paragraph.add_run(text)
    set_font(run, size=10.0)


def add_code_block(document: Document, code_text: str) -> None:
    table = document.add_table(rows=1, cols=1)
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.8)
    set_cell_shading(cell, "F8F9FA")
    set_cell_border(cell, color="CCCCCC")
    cell.text = ""
    for line in code_text.split("\n"):
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        run = p.add_run(line if line.strip() else " ")
        set_font(run, name="Consolas", size=8.0)
    document.add_paragraph().paragraph_format.space_after = Pt(6)


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(10)
    footer = section.footer
    footer_p = footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_p.paragraph_format.space_before = Pt(4)
    run = footer_p.add_run("EC24313 Assessment 2 | BIT Mesra | Anurag Tiwari (BTECH/10142/24) | ")
    set_font(run, size=8.5)
    add_page_number(footer_p)


def main() -> None:
    metrics = json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8"))
    features = pd.read_csv(OUTPUTS / "mfcc_features.csv")
    manifest = pd.read_csv(OUTPUTS / "dataset_manifest.csv")
    group_review = pd.read_csv(OUTPUTS / "speaker_group_review.csv")
    benchmarks = pd.read_csv(OUTPUTS / "model_comparison.csv")
    code_text = CODE_FILE.read_text(encoding="utf-8")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    boys = int((features["class_name"] == "boys").sum())
    girls = int((features["class_name"] == "girls").sum())
    total = len(features)
    included = int((manifest["status"] == "included").sum())
    matrix = metrics["confusion_matrix_rows_actual_columns_predicted"]

    document = Document()
    configure_document(document)

    # Cover Page
    for _ in range(2):
        document.add_paragraph()
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Department of Electronics and Communication Engineering")
    set_font(run, size=15, bold=True)
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Birla Institute of Technology Mesra, Ranchi")
    set_font(run, size=13, bold=True)
    
    document.add_paragraph()
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(25)
    title.paragraph_format.space_after = Pt(8)
    run = title.add_run("Assessment 2: Voice Gender Classification")
    set_font(run, name="Aptos Display", size=24, bold=True)
    
    sub = document.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Mel-Frequency Cepstral Coefficients (13 MFCCs) & Support Vector Machine")
    set_font(run, size=12, italic=True)

    for _ in range(4):
        document.add_paragraph()

    meta = [
        "Course Code: EC24313",
        "Course Name: Artificial Intelligence and Machine Learning",
        "Student Name: Anurag Tiwari",
        "Roll Number: BTECH/10142/24",
        "Dataset: Voice Dataset for Assignment 2 (148 Audio Recordings)",
        "GitHub Repository: https://github.com/Anuragluck/Voice-Gender-Classification-MFCC",
        "Submission Date: November 2026",
    ]
    for line in meta:
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(line)
        set_font(run, size=10.5)

    document.add_page_break()

    # Executive Summary
    add_heading(document, "Executive Summary", level=1)
    add_body(
        document,
        f"This report presents the end-to-end implementation and evaluation of a voice-based gender classification system for Assessment 2 in EC24313 (AIML). "
        f"The dataset consists of 148 student audio recordings across two ground-truth folders: Class 1 (Boys, 129 recordings) and Class 2 (Girls, 19 recordings). "
        f"Each utterance was preprocessed using mono conversion, 16 kHz resampling, peak normalization, and leading/trailing silence trimming at a -30 dB threshold, directly extending the signal-processing methodology established in Assessment 1. "
        f"Exactly 13 Mel-Frequency Cepstral Coefficients (MFCCs) were extracted per frame and temporally summarized via mean and standard deviation, yielding a 26-dimensional feature vector per sample. "
        f"To prevent data leakage caused by paired 'BIT' and 'Name' utterances from the same speaker, evaluation was conducted using 5-Fold Stratified Group Cross-Validation. "
        f"A class-balanced Radial Basis Function Support Vector Machine (RBF SVM) achieved an overall accuracy of {metrics['accuracy']*100:.2f}%, sensitivity of {metrics['sensitivity_recall_girls']*100:.2f}% on girls, precision of {metrics['precision_girls']*100:.2f}%, and an ROC-AUC of {metrics['roc_auc_girls']:.3f}."
    )

    # 1. Objective
    add_heading(document, "1. Objective and Problem Formulation", level=1)
    add_body(
        document,
        "The objective of this assignment is to design, implement, and rigorously evaluate a supervised machine learning pipeline capable of classifying human voice recordings into Class 1 (boys) or Class 2 (girls). "
        "The required deliverables comprise: (1) dataset ingestion and analysis, (2) acoustic feature extraction using 13 MFCCs and scatter plot visualization, (3) model development using an appropriate machine learning classifier (NN/SVM/RF/CNN), and (4) comprehensive evaluation using a Confusion Matrix, Accuracy, Sensitivity, Precision, and ROC-AUC curve."
    )

    # 2. Dataset and Preprocessing
    add_heading(document, "2. Dataset Exploration, Ingestion, and Silence Trimming", level=1)
    add_body(
        document,
        "The dataset was compiled from the class cohort of Assignment 1. Ground-truth labels are established directly by the directory structure: "
        "'VOICE DATASET BOYS' maps to Class 1, and 'VOICE DATASET GIRLS' maps to Class 2. "
        "The archive displays substantial file format diversity, containing 115 WAV files, 27 MP3 files, 1 M4A file, 1 MP4 file, and 4 extensionless audio streams. "
        "A robust two-tier audio decoding pipeline was engineered: PySoundFile is utilized for native PCM WAV streams, with an automated fallback to FFmpeg for encoded AAC, MP4, and compressed formats. All 148 recordings were successfully ingested without data loss."
    )
    add_body(
        document,
        "In accordance with Assessment 1, where silent and unvoiced intervals were detected at -30 dB, every audio signal underwent standardized preprocessing: "
        "(a) downmixing multi-channel signals to single-channel mono, (b) resampling to a uniform sampling frequency fs = 16,000 Hz, (c) peak amplitude normalization to unity scale [-1.0, 1.0], and (d) trimming silence below a 30 dB relative threshold."
    )

    add_table(
        document,
        ["Class Name", "Class Label", "Usable Recordings", "Percentage Share", "Unique Speaker Groups"],
        [
            ["Boys", "Class 1", str(boys), f"{boys/total*100:.1f}%", "65 groups"],
            ["Girls", "Class 2", str(girls), f"{girls/total*100:.1f}%", "11 groups"],
            ["Total Cohort", "Classes 1 & 2", str(total), "100.0%", "76 groups"],
        ],
        [1.5, 1.1, 1.4, 1.3, 1.5],
    )

    document.add_picture(str(OUTPUTS / "01_class_distribution.png"), width=Inches(5.6))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "Figure 1. Dataset Class Distribution illustrating severe class imbalance (87.2% Boys vs 12.8% Girls).")

    # 3. MFCC Extraction
    add_heading(document, "3. Acoustic Feature Extraction (13 MFCCs) and Visualizations", level=1)
    add_body(
        document,
        "Mel-Frequency Cepstral Coefficients (MFCCs) represent the dominant acoustic feature representation for speech and speaker characterization. "
        "They approximate human auditory frequency response, which exhibits linear frequency resolution below 1 kHz and logarithmic resolution above 1 kHz. "
        "The extraction process proceeds through five mathematical stages:"
    )
    add_body(
        document,
        "1. Framing and Windowing: The continuous speech signal is divided into overlapping frames using a 512-sample FFT window (32 ms duration at 16 kHz) and a 256-sample hop length (16 ms step) weighted by a Hamming window.\n"
        "2. Short-Time Fourier Transform (STFT): Converts each frame into its discrete frequency power spectrum.\n"
        "3. Mel Filterbank Integration: The power spectrum is passed through triangular overlapping bandpass filters spaced along the Mel scale: m = 2595 * log10(1 + f / 700).\n"
        "4. Logarithmic Compression: Computes the logarithm of the filterbank energies, mirroring human perceived loudness.\n"
        "5. Discrete Cosine Transform (DCT-II): Decorrelates the log-filterbank energies to yield the first 13 cepstral coefficients."
    )
    add_body(
        document,
        "Because spoken utterances vary in temporal duration (e.g., 'BIT' is typically 0.8-1.5 s while full student names span 1.2-3.5 s), "
        "frame-level MFCC matrices must be transformed into fixed-dimension vectors. For each of the 13 coefficients, the mean (capturing stationary vocal tract resonance) and standard deviation (capturing pitch dynamics and expressive variation) were calculated across frames. "
        "This yields a rich, fixed-length 26-dimensional representation per audio recording."
    )

    document.add_picture(str(OUTPUTS / "02_mfcc_pca_scatter.png"), width=Inches(5.5))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "Figure 2. PCA 2D Scatter Plot of the 13-MFCC summary features (PC1 vs PC2) demonstrating gender clustering.")

    document.add_picture(str(OUTPUTS / "02b_mfcc_scatter_direct.png"), width=Inches(5.5))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "Figure 2b. Direct Acoustic Scatter Plot of MFCC Coefficient 2 Mean versus MFCC Coefficient 3 Mean.")

    # 4. Model Architecture & Cross-Validation
    add_heading(document, "4. Machine Learning Model Architecture & Experimental Design", level=1)
    add_body(
        document,
        "Given the small sample size (N = 148) and low feature dimensionality (D = 26), Support Vector Machines (SVM) with a Radial Basis Function (RBF) kernel offer decisive mathematical advantages over deep convolutional neural networks (CNNs), which are prone to severe overfitting on small datasets."
    )
    add_body(
        document,
        "To address the severe 6.8:1 class imbalance, cost-sensitive learning was introduced via balanced class weighting: w_j = N / (2 * N_j). "
        "Under this weighting scheme, misclassifying a minority girl sample incurs a penalty 6.79 times higher than misclassifying a majority boy sample, preventing the decision boundary from collapsing into a trivial majority-class predictor."
    )
    add_body(
        document,
        "Critical Experimental Design: Speaker Leakage Prevention. Each student submitted two distinct recordings ('BIT' and 'Name'). If a student's 'BIT' recording was placed in the training fold while their 'Name' recording was assigned to the test fold, the classifier would inadvertently recognize the unique speaker's identity rather than learning general gender characteristics. "
        "To enforce rigorous academic integrity, conservative speaker IDs were derived from filenames, and a 5-Fold Stratified Group K-Fold split was enforced. Paired recordings from any individual student remained strictly within the same fold."
    )

    add_table(
        document,
        ["System Parameter", "Configuration / Value", "Design Rationale"],
        [
            ["Classifier", "Support Vector Machine (SVC)", "Optimal margin separation for small acoustic datasets"],
            ["Kernel Function", "Radial Basis Function (RBF)", "K(x, x') = exp(-gamma * ||x - x'||^2) for non-linear boundary"],
            ["Regularization (C)", "C = 4.0", "Tuned margin penalty balancing slack variables and margin width"],
            ["Class Weighting", "class_weight = 'balanced'", "Compensates for 129:19 imbalance (w_girls = 3.89, w_boys = 0.57)"],
            ["Cross-Validation", "5-Fold Stratified Group K-Fold", "Guarantees zero speaker identity leakage across train/test folds"],
            ["Feature Scaler", "StandardScaler", "Z-score normalization fit strictly on training folds"],
        ],
        [1.8, 2.3, 2.7],
    )

    # 5. Results & Evaluation Metrics
    add_heading(document, "5. Model Performance Evaluation and Comparative Results", level=1)
    add_body(
        document,
        "Model performance was evaluated strictly on out-of-fold predictions across the full 148-sample dataset. "
        "Because Class 2 (girls) is the minority class, sensitivity (recall) and precision for girls are the critical operational metrics, reported alongside overall accuracy and ROC-AUC."
    )

    add_table(
        document,
        ["Evaluation Metric", "Formula", "Observed Value", "Operational Interpretation"],
        [
            ["Accuracy", "(TP + TN) / Total", f"{metrics['accuracy']*100:.2f}%", "136 correct classifications out of 148 utterances"],
            ["Sensitivity (Recall)", "TP / (TP + FN)", f"{metrics['sensitivity_recall_girls']*100:.2f}%", "9 out of 19 girls correctly identified"],
            ["Precision (Girls)", "TP / (TP + FP)", f"{metrics['precision_girls']*100:.2f}%", "9 of 11 predicted girls were truly female (low false alarm)"],
            ["Specificity (Boys)", "TN / (TN + FP)", "98.45%", "127 of 129 boys correctly identified"],
            ["ROC-AUC", "Integral of TPR(FPR)", f"{metrics['roc_auc_girls']:.3f}", "High discriminative ranking ability across thresholds"],
        ],
        [1.5, 1.4, 1.1, 2.8],
    )

    document.add_picture(str(OUTPUTS / "03_confusion_matrix.png"), width=Inches(4.8))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "Figure 3. Out-of-fold Confusion Matrix. True Classes along rows; Predicted Classes along columns.")

    document.add_picture(str(OUTPUTS / "04_roc_curve.png"), width=Inches(5.0))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "Figure 4. Receiver Operating Characteristic (ROC) Curve for Girls detection (AUC = 0.910).")

    add_heading(document, "5.1 Comparative Benchmark of Machine Learning Classifiers", level=2)
    add_body(
        document,
        "In compliance with the assignment syllabus (which lists NN/SVM/RF/CNN options), we benchmarked the primary RBF SVM against alternative classifiers under the exact same 5-fold speaker-grouped cross-validation protocol:"
    )

    comp_rows = []
    for _, row in benchmarks.iterrows():
        comp_rows.append([
            str(row["model_name"]),
            f"{row['accuracy']*100:.2f}%",
            f"{row['girls_sensitivity_recall']*100:.2f}%",
            f"{row['girls_precision']*100:.2f}%",
            f"{row['girls_f1']:.4f}",
            f"{row['roc_auc']:.4f}",
        ])

    add_table(
        document,
        ["Model Architecture", "Accuracy", "Girls Recall", "Girls Prec.", "Girls F1", "ROC-AUC"],
        comp_rows,
        [2.3, 0.9, 0.9, 0.9, 0.9, 0.9],
    )

    # 6. Discussion and Dataset Limitations
    add_heading(document, "6. Discussion, Limitations, and Dataset Anomalies", level=1)
    add_body(
        document,
        "The empirical results confirm that 13 MFCC summary features capture significant spectral envelope and formant characteristics necessary for gender classification. "
        "However, two key real-world challenges must be highlighted:"
    )
    add_body(
        document,
        "1. Class Imbalance Impact: Despite class-balanced weighting, the limited representation of girls (19 samples across 11 speakers) restricts the model's ability to capture the full intra-class acoustic variance of female speech. While precision is high (81.82%), sensitivity sits at 47.37%, indicating that borderline female voices with deeper pitch are occasionally misclassified as boys.\n"
        "2. Ground-Truth Anomaly ('Shreya' Recordings): During speaker group manifest inspection, a data entry inconsistency was identified in the source archive: 'Shreya_Name.mp3' was stored in VOICE DATASET BOYS, while 'Shreya_BIT.mp3' was stored in VOICE DATASET GIRLS. Following instructions, the folder labels were retained without tampering, and both utterances were assigned to Fold 4. The model classified both utterances consistently based on their acoustic features.\n"
        "3. Verification on Student's Own Audio: Both recordings submitted by Anurag Tiwari (AnuragTiwari_BIT.wav and AnuragTiwari_Name.wav) were evaluated in held-out Fold 4. Both samples were correctly classified as Class 1 (Boys) with confident negative decision margins (-1.26 and -0.52)."
    )

    # 7. Conclusion
    add_heading(document, "7. Conclusion", level=1)
    add_body(
        document,
        "Assessment 2 has been successfully completed. All 148 recordings were decoded, normalized, and trimmed using a -30 dB silence threshold consistent with Assessment 1. "
        "13 Mel-Frequency Cepstral Coefficients were extracted and pooled into a 26-dimensional summary representation. "
        "A class-balanced RBF Support Vector Machine combined with speaker-grouped 5-fold cross-validation demonstrated high discriminative performance (91.89% Accuracy, 0.910 ROC-AUC) without suffering from speaker leakage. "
        "The complete, fully reproducible implementation is delivered in the accompanying code files and Jupyter Notebook."
    )

    # 8. Complete Model Algorithm Code
    add_heading(document, "8. Complete Model Algorithm and Source Code", level=1)
    add_body(
        document,
        "The complete Python implementation algorithm (src/run_assignment2.py) is reproduced below, fulfilling the requirement to provide the complete code alongside output plots and results:"
    )
    add_code_block(document, code_text)

    document.save(REPORT_PATH)
    print(f"Report DOCX successfully generated at: {REPORT_PATH}")


if __name__ == "__main__":
    main()
