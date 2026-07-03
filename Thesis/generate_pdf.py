import os
from fpdf import FPDF
import re

class ThesisPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, 'Automated Egg Embryo Blood Vessel Segmentation - Thesis', 0, 0, 'C')
            self.ln(10)
            
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, num, label):
        self.add_page()
        self.set_font('helvetica', 'B', 18)
        self.cell(0, 15, f'Chapter {num}: {label}', 0, 1, 'L')
        self.ln(5)
        
    def chapter_body(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            txt = f.read()
            
        # Basic LaTeX stripping and cleaning
        txt = re.sub(r'\\chapter{.*?}', '', txt)
        txt = re.sub(r'\\label{.*?}', '', txt)
        txt = re.sub(r'\\parencite{.*?}', ' [Citation] ', txt)
        txt = re.sub(r'\\textcite{.*?}', ' [Citation] ', txt)
        txt = re.sub(r'\\ref{.*?}', '[Ref]', txt)
        txt = re.sub(r'\\textbf{(.*?)}', r'\1', txt)
        txt = re.sub(r'\\textit{(.*?)}', r'\1', txt)
        txt = re.sub(r'\\begin{.*?}', '', txt)
        txt = re.sub(r'\\end{.*?}', '', txt)
        txt = re.sub(r'\\item', '- ', txt)
        txt = re.sub(r'\$.*?\$', '[Eq.]', txt) # remove inline math
        txt = re.sub(r'\\includegraphics.*?{.*?}', '', txt)
        txt = re.sub(r'\\caption{.*?}', '', txt)
        txt = re.sub(r'\\begin{figure}.*?\\end{figure}', '', txt, flags=re.DOTALL)
        txt = re.sub(r'\\begin{equation}.*?\\end{equation}', '[Equation]', txt, flags=re.DOTALL)
        txt = re.sub(r'\\begin{align}.*?\\end{align}', '[Equation]', txt, flags=re.DOTALL)
        txt = re.sub(r'\\begin{table}.*?\\end{table}', '[Table Placeholder]', txt, flags=re.DOTALL)
        txt = re.sub(r'\\begin{algorithm}.*?\\end{algorithm}', '[Algorithm Placeholder]', txt, flags=re.DOTALL)
        txt = txt.replace('\\', '')
        txt = txt.replace('\u2014', '-')  # em-dash
        txt = txt.replace('\u2013', '-')  # en-dash
        txt = txt.replace('\u2019', "'")  # right single quote
        txt = txt.replace('\u2018', "'")  # left single quote
        txt = txt.replace('\u201c', '"')  # left double quote
        txt = txt.replace('\u201d', '"')  # right double quote
        txt = txt.replace('\mu g', 'ug')
        txt = txt.replace('\circ', ' degrees ')
        txt = txt.replace('\pm', '+/-')
        
        # Split by section
        parts = txt.split('section{')
        
        for i, part in enumerate(parts):
            if i == 0:
                self.set_font('helvetica', '', 11)
                paragraphs = part.strip().split('\n\n')
                for p in paragraphs:
                    p = p.strip()
                    if p:
                        self.multi_cell(0, 7, p)
                        self.ln(3)
            else:
                idx = part.find('}')
                if idx != -1:
                    section_title = part[:idx]
                    body = part[idx+1:].strip()
                else:
                    section_title = "Section"
                    body = part.strip()
                
                self.ln(4)
                self.set_font('helvetica', 'B', 13)
                self.cell(0, 10, section_title, 0, 1, 'L')
                self.ln(2)
                
                self.set_font('helvetica', '', 11)
                paragraphs = body.split('\n\n')
                for p in paragraphs:
                    p = p.strip()
                    if p:
                        self.multi_cell(0, 7, p)
                        self.ln(3)
        self.ln(5)

def generate_pdf():
    pdf = ThesisPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 20, '', 0, 1, 'C')
    pdf.multi_cell(0, 10, 'Automated Angiography Blood Vessel Segmentation and\nMorphological Analysis using Deep Learning Architectures', 0, 'C')
    pdf.ln(15)
    pdf.set_font('helvetica', 'I', 13)
    pdf.cell(0, 10, 'A Thesis Submitted in Partial Fulfillment of the Requirements', 0, 1, 'C')
    pdf.cell(0, 10, 'for the Degree of', 0, 1, 'C')
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, 'Integrated Dual Degree (IDD)', 0, 1, 'C')
    pdf.ln(25)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'By:', 0, 1, 'C')
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 8, 'Seemant', 0, 1, 'C')
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 8, 'Roll No.: 21024014', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Under the supervision of:', 0, 1, 'C')
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 8, 'Dr. Pradip Paik', 0, 1, 'C')
    pdf.ln(25)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 8, 'Department of Biomedical Engineering', 0, 1, 'C')
    pdf.cell(0, 8, 'Indian Institute of Technology (BHU)', 0, 1, 'C')
    pdf.cell(0, 8, 'Varanasi, India', 0, 1, 'C')
    pdf.cell(0, 8, '2026', 0, 1, 'C')
    
    # Chapters
    chapters = [
        (1, "Introduction", "chapters/1_Introduction.tex"),
        (2, "Literature Review", "chapters/2_Literature_Review.tex"),
        (3, "System Model and Preliminaries", "chapters/3_System_Model.tex"),
        (4, "Proposed Methodology", "chapters/4_Methodology.tex"),
        (5, "Experimental Design", "chapters/5_Experimental_Design.tex"),
        (6, "Results and Discussion", "chapters/6_Results_and_Discussion.tex"),
        (7, "Conclusion and Future Work", "chapters/7_Conclusion.tex")
    ]
    
    for num, title, path in chapters:
        pdf.chapter_title(num, title)
        pdf.chapter_body(path)
        
        # Add corresponding figures after chapter text
        if num == 1:
            pdf.ln(5)
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, 'Figure 1.1: Representative example from the CAM angiography dataset', 0, 1, 'L')
            if os.path.exists('images/sample_rgb_501.jpg'):
                pdf.image('images/sample_rgb_501.jpg', w=80, h=80)
            if os.path.exists('images/sample_mask_501.jpg'):
                pdf.image('images/sample_mask_501.jpg', w=80, h=80)
                
        elif num == 5:
            pdf.ln(5)
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, 'Table 5.1: Summary of Hyperparameter Configuration', 0, 1, 'L')
            
        elif num == 6:
            pdf.ln(5)
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, 'Figure 6.1: Combined Training Metrics Plot', 0, 1, 'L')
            if os.path.exists('images/training_metrics_plot.png'):
                pdf.image('images/training_metrics_plot.png', w=160)
            
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, 'Figure 6.2: Detailed UNet++ (ResNet50) Training Curves', 0, 1, 'L')
            if os.path.exists('images/unetpp_training_curve.png'):
                pdf.image('images/unetpp_training_curve.png', w=150)
                
            pdf.ln(5)
            pdf.cell(0, 8, 'Figure 6.3: Detailed DeepLabV3+ Training Curves', 0, 1, 'L')
            if os.path.exists('images/deeplabv3plus_training_curve.png'):
                pdf.image('images/deeplabv3plus_training_curve.png', w=150)
            
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, 'Figure 6.4: Qualitative visual output for test image 501', 0, 1, 'L')
            if os.path.exists('images/single_prediction_501.png'):
                pdf.image('images/single_prediction_501.png', w=150)
            
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, 'Figure 6.5: CAM Assay Angiogenesis growth dynamics over time', 0, 1, 'L')
            if os.path.exists('images/vessel_growth_plots.png'):
                pdf.image('images/vessel_growth_plots.png', w=160)
                
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, 'Figure 6.6: Pearson correlation scatter plots comparing GT and predicted topology', 0, 1, 'L')
            if os.path.exists('images/pearson_correlation_plots.png'):
                pdf.image('images/pearson_correlation_plots.png', w=160)
                
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, 'Figure 6.7: Spearman rank correlation heatmaps', 0, 1, 'L')
            if os.path.exists('images/spearman_correlation_heatmaps.png'):
                pdf.image('images/spearman_correlation_heatmaps.png', w=150)
                
            pdf.ln(5)
            pdf.cell(0, 8, 'Figure 6.8: Statistical t-test box plot distributions', 0, 1, 'L')
            if os.path.exists('images/statistical_ttest_boxplots.png'):
                pdf.image('images/statistical_ttest_boxplots.png', w=150)
                
    pdf.output("Final_Thesis_Document.pdf")
    print("PDF successfully generated as Final_Thesis_Document.pdf")

if __name__ == '__main__':
    generate_pdf()

