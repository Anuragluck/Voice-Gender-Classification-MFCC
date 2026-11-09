"""Generate the submission-ready PDF report for EC24313 Assessment 2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_PATH = REPORT_DIR / "Anurag_Tiwari_Assessment2_Report.pdf"
CODE_FILE = PROJECT_ROOT / "src" / "run_assignment2.py"

BLUE = colors.HexColor("#1F4E79")
NAVY = colors.HexColor("#0D233A")
LIGHT_BLUE = colors.HexColor("#DCE6F1")
LIGHT_GRAY = colors.HexColor("#F4F5F7")
DARK_GRAY = colors.HexColor("#2B2B2B")
BORDER_GRAY = colors.HexColor("#D9D9D9")


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    scale = min(max_width / width, max_height / height)
    img = Image(str(path), width=width * scale, height=height * scale)
    img.hAlign = "CENTER"
    return img


def header_footer(canvas, document) -> None:
    canvas.saveState()
    # Running Header
    canvas.setStrokeColor(BORDER_GRAY)
    canvas.setLineWidth(0.5)
    canvas.line(0.75 * inch, 10.45 * inch, 7.75 * inch, 10.45 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(0.75 * inch, 10.52 * inch, "Birla Institute of Technology Mesra | EC24313 AIML")
    canvas.drawRightString(7.75 * inch, 10.52 * inch, "Assessment 2: Voice Gender Classification")

    # Running Footer
    canvas.line(0.75 * inch, 0.55 * inch, 7.75 * inch, 0.55 * inch)
    canvas.drawString(0.75 * inch, 0.40 * inch, "Student: Anurag Tiwari | Roll No: BTECH/10142/24")
    canvas.drawRightString(7.75 * inch, 0.40 * inch, f"Page {document.page}")
    canvas.restoreState()


def create_table(data, widths, header=True) -> Table:
    cell_style = ParagraphStyle("CellLeft", fontName="Helvetica", fontSize=8.0, leading=10.0, textColor=DARK_GRAY)
    cell_center = ParagraphStyle("CellCenter", fontName="Helvetica", fontSize=8.0, leading=10.0, alignment=TA_CENTER, textColor=DARK_GRAY)
    header_style = ParagraphStyle("CellHdr", fontName="Helvetica-Bold", fontSize=8.0, leading=10.0, alignment=TA_CENTER, textColor=colors.white)

    formatted_data = []
    for row_idx, row in enumerate(data):
        formatted_row = []
        is_hdr = (row_idx == 0 and header)
        for col_idx, cell in enumerate(row):
            if isinstance(cell, str):
                if is_hdr:
                    formatted_row.append(Paragraph(cell, header_style))
                elif col_idx == 0:
                    formatted_row.append(Paragraph(cell, cell_style))
                else:
                    formatted_row.append(Paragraph(cell, cell_center))
            else:
                formatted_row.append(cell)
        formatted_data.append(formatted_row)

    t = Table(formatted_data, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
            ]
        )
    t.setStyle(TableStyle(commands))
    return t


def main() -> None:
    metrics = json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8"))
    features = pd.read_csv(OUTPUTS / "mfcc_features.csv")
    manifest = pd.read_csv(OUTPUTS / "dataset_manifest.csv")
    benchmarks = pd.read_csv(OUTPUTS / "model_comparison.csv")
    code_text = CODE_FILE.read_text(encoding="utf-8")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    boys = int((features["class_name"] == "boys").sum())
    girls = int((features["class_name"] == "girls").sum())
    total = len(features)
    matrix = metrics["confusion_matrix_rows_actual_columns_predicted"]

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Assessment 2 Voice Gender Classification - Anurag Tiwari (BTECH/10142/24)",
        author="Anurag Tiwari (BTECH/10142/24)",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("CoverDept", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=14, leading=18, alignment=TA_CENTER, textColor=BLUE))
    styles.add(ParagraphStyle("CoverInst", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, leading=16, alignment=TA_CENTER, textColor=DARK_GRAY))
    styles.add(ParagraphStyle("CoverTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=23, leading=28, alignment=TA_CENTER, textColor=NAVY, spaceBefore=20, spaceAfter=8))
    styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=12, leading=16, alignment=TA_CENTER, textColor=DARK_GRAY, spaceAfter=20))
    styles.add(ParagraphStyle("CoverMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5, leading=15, alignment=TA_CENTER, textColor=DARK_GRAY))
    styles.add(ParagraphStyle("SecHeading", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12.5, leading=15, textColor=BLUE, spaceBefore=10, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle("SubHeading", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=DARK_GRAY, spaceBefore=7, spaceAfter=3, keepWithNext=True))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=9.2, leading=12.5, textColor=DARK_GRAY, spaceAfter=5, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=8.0, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=7))
    styles.add(ParagraphStyle("CodeStyle", parent=styles["Normal"], fontName="Courier", fontSize=6.5, leading=8.0, textColor=colors.HexColor("#1A1A1A"), backColor=LIGHT_GRAY))

    story = []

    # ================= PAGE 1: COVER PAGE =================
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("Department of Electronics and Communication Engineering", styles["CoverDept"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Birla Institute of Technology Mesra, Ranchi", styles["CoverInst"]))
    story.append(Spacer(1, 0.4 * inch))
    story.append(HRFlowable(width="80%", thickness=1.5, color=BLUE, spaceBefore=10, spaceAfter=15))
    story.append(Paragraph("Assessment 2: Voice Gender Classification", styles["CoverTitle"]))
    story.append(Paragraph("Acoustic Feature Extraction (13 MFCCs), Support Vector Machine Classifier & Performance Evaluation", styles["CoverSub"]))
    story.append(HRFlowable(width="80%", thickness=1.5, color=BLUE, spaceBefore=10, spaceAfter=20))
    story.append(Spacer(1, 1.4 * inch))

    meta_items = [
        "<b>Course Code:</b> EC24313",
        "<b>Course Title:</b> Artificial Intelligence and Machine Learning",
        "<b>Student Name:</b> Anurag Tiwari",
        "<b>Roll Number:</b> BTECH/10142/24",
        "<b>Dataset:</b> Voice Dataset for Assignment 2 (148 Spoken Utterances)",
        "<b>GitHub Repository:</b> <font color='#1F4E79'><u>https://github.com/Anuragluck/Voice-Gender-Classification-MFCC</u></font>",
        "<b>Date of Submission:</b> November 2026",
    ]
    for item in meta_items:
        story.append(Paragraph(item, styles["CoverMeta"]))
        story.append(Spacer(1, 3))

    story.append(PageBreak())

    # ================= PAGE 2: EXECUTIVE SUMMARY & DATASET =================
    story.append(Paragraph("Executive Summary", styles["SecHeading"]))
    story.append(
        Paragraph(
            f"This study develops an automated voice gender recognition system for Assessment 2 in EC24313 (AIML). "
            f"The dataset comprises 148 audio recordings collected from students, structured into Class 1 (Boys, 129 recordings) and Class 2 (Girls, 19 recordings). "
            f"Extending the preprocessing established in Assessment 1, each audio file was standardized to 16 kHz mono, peak-normalized, and trimmed at a -30 dB silence threshold. "
            f"Acoustic feature extraction captured exactly 13 Mel-Frequency Cepstral Coefficients (MFCCs) per frame, summarized into a 26-dimensional representation via mean and standard deviation. "
            f"To prevent data leakage caused by paired 'BIT' and 'Name' recordings from the same speaker, evaluation was conducted via 5-Fold Stratified Group K-Fold cross-validation. "
            f"A class-balanced RBF Support Vector Machine achieved <b>{metrics['accuracy']*100:.2f}% accuracy</b>, <b>{metrics['sensitivity_recall_girls']*100:.2f}% girls' sensitivity (recall)</b>, <b>{metrics['precision_girls']*100:.2f}% precision</b>, and an <b>ROC-AUC of {metrics['roc_auc_girls']:.3f}</b>.",
            styles["Body"],
        )
    )

    story.append(Paragraph("1. Objective and Problem Statement", styles["SecHeading"]))
    story.append(
        Paragraph(
            "The objective is to classify speech signals into Class 1 (boys) or Class 2 (girls) according to the departmental assignment brief: "
            "(1) Ingest and analyze the class voice dataset, (2) extract 13 MFCC acoustic coefficients and visualize sample distributions via 2D scatter plots, "
            "(3) construct a machine learning classifier with appropriate class balancing, and (4) evaluate performance using a Confusion Matrix, Accuracy, Sensitivity, Precision, and ROC curve.",
            styles["Body"],
        )
    )

    story.append(Paragraph("2. Dataset Exploration, Ingestion, and Silence Trimming", styles["SecHeading"]))
    story.append(
        Paragraph(
            "The archive contains 148 recordings across diverse audio formats: 115 WAV, 27 MP3, 1 M4A, 1 MP4, and 4 extensionless files. "
            "A dual-tier decoder was implemented: PySoundFile handles standard PCM WAV files, while FFmpeg decodes compressed audio streams. All 148 recordings were decoded without exclusion.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Direct continuity with Assessment 1 was maintained: each recording was converted to mono, resampled to 16 kHz, normalized to [-1.0, 1.0], and trimmed at leading/trailing edges below a -30 dB threshold using <i>librosa.effects.trim(top_db=30)</i>.",
            styles["Body"],
        )
    )

    story.append(
        create_table(
            [
                ["Class Name", "Class Label", "Recordings", "Proportion", "Unique Speakers"],
                ["Boys", "Class 1", str(boys), f"{boys/total*100:.1f}%", "65 groups"],
                ["Girls", "Class 2", str(girls), f"{girls/total*100:.1f}%", "11 groups"],
                ["Cohort Total", "Classes 1 & 2", str(total), "100.0%", "76 groups"],
            ],
            [1.5 * inch, 1.0 * inch, 1.2 * inch, 1.1 * inch, 1.7 * inch],
        )
    )
    story.append(Spacer(1, 4))
    story.append(scaled_image(OUTPUTS / "01_class_distribution.png", 5.2 * inch, 2.7 * inch))
    story.append(Paragraph("Figure 1. Class distribution of usable recordings showing severe imbalance (87.2% boys vs 12.8% girls).", styles["Caption"]))

    story.append(PageBreak())

    # ================= PAGE 3: MFCC FEATURE EXTRACTION =================
    story.append(Paragraph("3. Acoustic Feature Extraction (13 MFCCs) & Visualizations", styles["SecHeading"]))
    story.append(
        Paragraph(
            "Mel-Frequency Cepstral Coefficients (MFCCs) capture the spectral envelope of human vocal tract resonances. "
            "The computational procedure consists of: (1) Pre-emphasis high-frequency boost, (2) Short-Time Fourier Transform framing with a 512-sample FFT window (32 ms) and 256-sample hop (16 ms) weighted by a Hamming window, "
            "(3) Mel-filterbank integration across triangular bandpass filters mapped via <i>m = 2595 * log10(1 + f / 700)</i>, (4) Logarithmic energy compression, and (5) Discrete Cosine Transform (DCT-II) decorrelation yielding 13 cepstral coefficients per frame.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "To convert variable-duration frame matrices into fixed-length classifier inputs, the <b>mean</b> and <b>standard deviation</b> were calculated across time frames for each of the 13 coefficients. "
            "This produces a 26-dimensional summary vector per utterance, effectively separating spectral envelope formants from vocal dynamics.",
            styles["Body"],
        )
    )

    story.append(scaled_image(OUTPUTS / "02_mfcc_pca_scatter.png", 4.9 * inch, 3.1 * inch))
    story.append(Paragraph("Figure 2. 2D Principal Component Analysis (PCA) projection of the 26-dimensional MFCC feature space.", styles["Caption"]))
    story.append(Spacer(1, 4))

    story.append(scaled_image(OUTPUTS / "02b_mfcc_scatter_direct.png", 4.9 * inch, 3.1 * inch))
    story.append(Paragraph("Figure 2b. Direct Acoustic Scatter Plot: MFCC 2 Mean (Spectral Tilt) vs MFCC 3 Mean (Low Formants).", styles["Caption"]))

    story.append(PageBreak())

    # ================= PAGE 4: MODEL & RESULTS =================
    story.append(Paragraph("4. Machine Learning Model Architecture & Experimental Design", styles["SecHeading"]))
    story.append(
        Paragraph(
            "For this small dataset (N = 148, D = 26), Support Vector Machines (SVM) with a Radial Basis Function (RBF) kernel provide superior generalization compared to deep CNNs, which overfit severely on small sample sizes. "
            "To counter the 6.8:1 class imbalance, cost-sensitive learning was applied with <i>class_weight='balanced'</i>: "
            "<i>w_j = N / (2 * N_j)</i>, giving <i>w_girls = 3.89</i> and <i>w_boys = 0.57</i>.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Leakage Prevention via Group Cross-Validation:</b> Students submitted paired 'BIT' and 'Name' recordings. "
            "Random train/test splits would place clips from the same speaker across both sets, allowing the model to recognize individual voices rather than gender. "
            "We enforced a 5-Fold Stratified Group K-Fold split on 76 speaker IDs, ensuring paired utterances never cross train/test boundaries.",
            styles["Body"],
        )
    )

    story.append(
        create_table(
            [
                ["Hyperparameter / Setting", "Assigned Value", "Design Rationale"],
                ["Classifier Architecture", "Support Vector Machine (SVC)", "Maximum-margin hyperplane in kernel Hilbert space"],
                ["Kernel Function", "Radial Basis Function (RBF)", "K(x, x') = exp(-gamma * ||x - x'||^2) for non-linear boundaries"],
                ["Regularization Penalty (C)", "C = 4.0", "Optimized trade-off between margin slack and boundary curvature"],
                ["Class Weighting Scheme", "class_weight = 'balanced'", "Weights errors inversely proportional to class frequency (w_g=3.89)"],
                ["Validation Scheme", "5-Fold Stratified Group K-Fold", "Strict zero speaker identity leakage across evaluation folds"],
            ],
            [1.9 * inch, 2.0 * inch, 2.6 * inch],
        )
    )

    story.append(Paragraph("5. Performance Evaluation and Results", styles["SecHeading"]))
    story.append(
        create_table(
            [
                ["Evaluation Metric", "Mathematical Formula", "Observed Value", "Operational Meaning"],
                ["Accuracy", "(TP + TN) / Total", f"{metrics['accuracy']*100:.2f}%", "136 correct predictions out of 148 utterances"],
                ["Sensitivity (Girls Recall)", "TP / (TP + FN)", f"{metrics['sensitivity_recall_girls']*100:.2f}%", "9 of 19 female recordings correctly recognized"],
                ["Precision (Girls)", "TP / (TP + FP)", f"{metrics['precision_girls']*100:.2f}%", "9 of 11 predicted girls were truly female (low false alarms)"],
                ["Specificity (Boys)", "TN / (TN + FP)", "98.45%", "127 of 129 male recordings correctly recognized"],
                ["ROC-AUC", "Integral of TPR(FPR)", f"{metrics['roc_auc_girls']:.3f}", "High discriminative ranking ability across thresholds"],
            ],
            [1.5 * inch, 1.4 * inch, 1.1 * inch, 2.5 * inch],
        )
    )
    story.append(Spacer(1, 4))
    story.append(scaled_image(OUTPUTS / "03_confusion_matrix.png", 4.4 * inch, 3.2 * inch))
    story.append(Paragraph("Figure 3. Out-of-fold Confusion Matrix (127 Boys and 9 Girls correctly identified).", styles["Caption"]))

    story.append(PageBreak())

    # ================= PAGE 5: COMPARISON & DISCUSSION =================
    story.append(Paragraph("5.1 Comparative Benchmark Across Classifier Families", styles["SecHeading"]))
    story.append(
        Paragraph(
            "To address the options specified in the syllabus (NN/SVM/RF/CNN), we benchmarked the primary RBF SVM against alternative machine learning classifiers under the exact same 5-fold speaker-grouped cross-validation protocol:",
            styles["Body"],
        )
    )

    bench_rows = [["Model Architecture", "Accuracy", "Girls Recall", "Girls Prec.", "Girls F1", "ROC-AUC"]]
    for _, row in benchmarks.iterrows():
        bench_rows.append([
            str(row["model_name"]),
            f"{row['accuracy']*100:.2f}%",
            f"{row['girls_sensitivity_recall']*100:.2f}%",
            f"{row['girls_precision']*100:.2f}%",
            f"{row['girls_f1']:.4f}",
            f"{row['roc_auc']:.4f}",
        ])
    story.append(create_table(bench_rows, [2.3 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.8 * inch]))
    story.append(Spacer(1, 4))

    story.append(scaled_image(OUTPUTS / "04_roc_curve.png", 4.3 * inch, 3.0 * inch))
    story.append(Paragraph("Figure 4. Receiver Operating Characteristic (ROC) Curve for Girls detection (AUC = 0.910).", styles["Caption"]))

    story.append(Paragraph("6. Discussion, Limitations, and Dataset Anomalies", styles["SecHeading"]))
    story.append(
        Paragraph(
            "<b>Class Imbalance Realities:</b> While overall accuracy is 91.89% and ROC-AUC is 0.910, sensitivity for girls is 47.37%. "
            "In highly skewed distributions (87.2% vs 12.8%), overall accuracy is dominated by the majority class. Precision for girls remains high at 81.82%, showing reliable positive detections.<br/>"
            "<b>Dataset Anomaly ('Shreya' Recordings):</b> Manifest auditing revealed that <i>Shreya_Name.mp3</i> was archived in the Boys folder while <i>Shreya_BIT.mp3</i> was archived in the Girls folder. "
            "The pipeline adhered to ground-truth folder labels without manual alteration and grouped both into Fold 4 for objective evaluation.<br/>"
            "<b>Verification of Student's Recordings:</b> Both utterances submitted by Anurag Tiwari (<i>AnuragTiwari_BIT.wav</i> and <i>AnuragTiwari_Name.wav</i>) were evaluated in held-out Fold 4 and correctly classified as Boys.",
            styles["Body"],
        )
    )

    story.append(Paragraph("7. Conclusion", styles["SecHeading"]))
    story.append(
        Paragraph(
            "Assessment 2 demonstrated the efficacy of 13 Mel-Frequency Cepstral Coefficients combined with class-balanced Support Vector Machines for voice gender recognition. "
            "Grouping speakers during cross-validation successfully prevented identity leakage, yielding honest and robust generalization metrics (91.89% accuracy, 0.910 ROC-AUC).",
            styles["Body"],
        )
    )

    story.append(PageBreak())

    # ================= PAGE 6+: CODE LISTING =================
    story.append(Paragraph("8. Complete Model Algorithm and Source Code", styles["SecHeading"]))
    story.append(
        Paragraph(
            "In compliance with the assignment instructions ('Prepare a final report along with your whole Model algorithm (Code)'), "
            "the complete, verified Python implementation script (<i>src/run_assignment2.py</i>) is reproduced below:",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(Preformatted(code_text, styles["CodeStyle"]))

    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=header_footer)
    print(f"Report PDF successfully generated at: {REPORT_PATH}")


if __name__ == "__main__":
    main()
