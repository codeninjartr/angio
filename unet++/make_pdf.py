import os
from fpdf import FPDF

class PresentationPDF(FPDF):
    def header(self):
        # We don't need a strict header on every page, just a subtle footer
        pass
        
    def footer(self):
        # Footer with slide number
        self.set_y(-15)
        self.set_font('helvetica', 'I', 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Slide {self.page_no()}', align='C')

    def add_title_slide(self):
        self.add_page()
        self.set_font('helvetica', 'B', 28)
        self.set_text_color(40, 40, 100)
        self.ln(50)
        self.cell(0, 20, 'Automated Retinal Blood Vessel Segmentation', align='C', ln=True)
        self.set_font('helvetica', 'I', 22)
        self.cell(0, 15, 'Using UNet++ Architecture', align='C', ln=True)
        self.ln(20)
        self.set_font('helvetica', '', 16)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, 'Thesis Defense Presentation', align='C', ln=True)
        self.cell(0, 10, 'IIT (BHU)', align='C', ln=True)

    def add_content_slide(self, title, points, img_path=None, img_w=200):
        self.add_page()
        
        # Title
        self.set_font('helvetica', 'B', 24)
        self.set_text_color(40, 40, 100)
        self.cell(0, 15, title, ln=True, align='L')
        self.line(10, 25, 287, 25)
        self.ln(10)
        
        # Points
        self.set_font('helvetica', '', 14)
        self.set_text_color(30, 30, 30)
        
        for point in points:
            if point.startswith("  "):  # Sub-bullet
                self.set_x(25)
                self.multi_cell(0, 8, "- " + point.strip())
            else:
                self.set_x(15)
                self.multi_cell(0, 10, "\x95 " + point)
            self.ln(2)

        # Optional Image
        if img_path and os.path.exists(img_path):
            self.ln(5)
            # Center the image
            x_pos = (297 - img_w) / 2
            self.image(img_path, x=x_pos, w=img_w)


def generate_presentation():
    pdf = PresentationPDF(orientation='L', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Title
    pdf.add_title_slide()

    # 2. Introduction
    pdf.add_content_slide(
        "Introduction & Motivation",
        [
            "Clinical Importance: Retinal blood vessels are key indicators for diagnosing cardiovascular",
            "  and ophthalmological diseases (e.g., Diabetic Retinopathy).",
            "The Challenge: Manual segmentation by clinicians is time-consuming, subjective, and",
            "  highly prone to human error.",
            "The Complexity: Retinal images feature fine, low-contrast micro-capillaries that are",
            "  incredibly difficult to extract precisely using traditional image processing.",
            "Objective: Develop an automated, highly precise deep learning model capable of",
            "  extracting the full vascular network from RGB fundus images."
        ]
    )

    # 3. Dataset
    pdf.add_content_slide(
        "The Dataset & Preprocessing",
        [
            "Source: Annotated clinical dataset of High-Resolution RGB Fundus images.",
            "Volume: 137 images split into training, validation, and testing sets.",
            "Ground Truth: Precise binary masks manually annotated by clinical experts.",
            "Preprocessing Pipeline:",
            "  Resize to unified dimensions of 256 x 256 for network compatibility.",
            "  Min-Max Normalization to scale pixel values into the [0, 1] range."
        ]
    )

    # 4. Methodology
    pdf.add_content_slide(
        "Methodology: UNet++ & Loss Function",
        [
            "Architecture: UNet++",
            "  A deeply supervised encoder-decoder network.",
            "  Features nested, dense skip pathways to bridge the semantic gap.",
            "  Vastly improved capture of both thick primary vessels and ultra-thin capillaries.",
            "Loss Function Formulation:",
            "  Loss = Binary Cross-Entropy (BCE) + Dice Loss",
            "  BCE: Ensures stable pixel-wise probability distributions.",
            "  Dice Loss: Solves severe class imbalance by directly maximizing Intersection over Union."
        ]
    )

    # 5. Experimental Setup
    pdf.add_content_slide(
        "Experimental Setup",
        [
            "Framework: Built in TensorFlow 2.x and Keras.",
            "Optimizer: Adam Optimizer for robust gradient updates.",
            "Learning Rate Scheduling:",
            "  ReduceLROnPlateau callback implemented to decay LR automatically when metrics stagnate.",
            "Metrics Tracked:",
            "  Dice Coefficient",
            "  Intersection over Union (IoU)",
            "  Accuracy, Precision, Recall"
        ]
    )

    # 6. Results - Metrics
    pdf.add_content_slide(
        "Results: Training Metrics",
        [
            "The model reached stable convergence over 26 epochs.",
            "Best Validation Dice Coefficient: ~0.926",
            "Best Validation IoU Score: ~0.901"
        ],
        img_path="outputs/training_metrics_plot.png",
        img_w=250
    )

    # 7. Results - Visual Inference
    pdf.add_content_slide(
        "Results: Visual Inference Output",
        [
            "Prediction on an unseen test sample (Image 501):",
            "Red overlay indicates predicted vessel boundaries matching the underlying image."
        ],
        img_path="outputs/single_prediction_501.png",
        img_w=250
    )

    # 8. Conclusion
    pdf.add_content_slide(
        "Conclusion & Future Work",
        [
            "Conclusion:",
            "  UNet++ coupled with a hybrid BCE+Dice loss achieves exceptional precision in",
            "  retinal vessel segmentation.",
            "  Successfully captures both macro and micro vascular structures.",
            "Future Work:",
            "  Integration of Spatial Attention Modules (Attention UNet++).",
            "  Multi-class classification to distinguish between Arteries and Veins.",
            "  Cross-dataset generalization testing (e.g., DRIVE or STARE datasets)."
        ]
    )

    # Save
    out_path = "outputs/Thesis_Presentation_UNet.pdf"
    pdf.output(out_path)
    print(f"[OK] Generated {out_path}")

if __name__ == "__main__":
    generate_presentation()
