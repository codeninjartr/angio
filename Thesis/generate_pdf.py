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
            
        # Very basic LaTeX strip/conversion
        txt = re.sub(r'\\chapter{.*?}', '', txt)
        txt = re.sub(r'\\label{.*?}', '', txt)
        txt = re.sub(r'\\parencite{.*?}', '[Citation]', txt)
        txt = re.sub(r'\\ref{.*?}', '[Ref]', txt)
        txt = re.sub(r'\\textbf{(.*?)}', r'\1', txt)
        txt = re.sub(r'\\textit{(.*?)}', r'\1', txt)
        txt = re.sub(r'\\begin{.*?}', '', txt)
        txt = re.sub(r'\\end{.*?}', '', txt)
        txt = re.sub(r'\\item', '-', txt)
        txt = re.sub(r'\$.*?\$', '[Equation]', txt) # remove inline math
        txt = re.sub(r'\\includegraphics.*?{.*?}', '[Figure Placeholder]', txt)
        txt = re.sub(r'\\caption{.*?}', '', txt)
        txt = re.sub(r'\\begin{figure}.*?\\end{figure}', '', txt, flags=re.DOTALL)
        txt = re.sub(r'\\begin{equation}.*?\\end{equation}', '[Equation]', txt, flags=re.DOTALL)
        txt = re.sub(r'\\begin{align}.*?\\end{align}', '[Equation]', txt, flags=re.DOTALL)
        txt = txt.replace('\\', '')
        txt = txt.replace('\u2014', '-')  # em-dash
        txt = txt.replace('\u2013', '-')  # en-dash
        txt = txt.replace('\u2019', "'")  # right single quote
        txt = txt.replace('\u2018', "'")  # left single quote
        txt = txt.replace('\u201c', '"')  # left double quote
        txt = txt.replace('\u201d', '"')  # right double quote
        
        # Split by section
        parts = txt.split('section{')
        
        for i, part in enumerate(parts):
            if i == 0:
                self.set_font('helvetica', '', 12)
                self.multi_cell(0, 8, part.strip())
            else:
                section_title = part[:part.find('}')]
                body = part[part.find('}')+1:].strip()
                
                self.ln(5)
                self.set_font('helvetica', 'B', 14)
                self.cell(0, 10, section_title, 0, 1, 'L')
                self.ln(2)
                
                self.set_font('helvetica', '', 12)
                # Handle paragraphs
                paragraphs = body.split('\n\n')
                for p in paragraphs:
                    p = p.strip()
                    if p:
                        self.multi_cell(0, 8, p)
                        self.ln(2)
        self.ln(10)

def generate_pdf():
    pdf = ThesisPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 18)
    pdf.cell(0, 30, '', 0, 1, 'C')
    pdf.cell(0, 10, 'Automated Angiography Blood Vessel Segmentation', 0, 1, 'C')
    pdf.cell(0, 10, 'and Morphological Analysis using Deep Learning Architectures', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('helvetica', 'I', 16)
    pdf.cell(0, 10, 'A Thesis Submitted in Partial Fulfillment of the Requirements', 0, 1, 'C')
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Integrated Dual Degree (IDD)', 0, 1, 'C')
    pdf.ln(30)
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, 'By: Seemant', 0, 1, 'C')
    pdf.cell(0, 10, 'Supervisor: Dr. Predeep Paik', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('helvetica', '', 14)
    pdf.cell(0, 10, 'Indian Institute of Technology (BHU)', 0, 1, 'C')
    pdf.cell(0, 10, 'Varanasi, India', 0, 1, 'C')
    
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
        
        # Add images directly to chapter 6
        if num == 6:
            pdf.ln(10)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Figure: Training Metrics', 0, 1, 'L')
            if os.path.exists('images/training_metrics_plot.png'):
                pdf.image('images/training_metrics_plot.png', w=160)
            
            pdf.add_page()
            pdf.cell(0, 10, 'Figure: Inference Output (Image 501)', 0, 1, 'L')
            if os.path.exists('images/single_prediction_501.png'):
                pdf.image('images/single_prediction_501.png', w=180)
    
    pdf.output("Final_Thesis_Document.pdf")
    print("PDF Generated!")

if __name__ == '__main__':
    generate_pdf()
