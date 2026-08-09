import pdfplumber
import pandas as pd
from langchain_text_splitter import RecursiveCharacterTextSplitter
class loader:
    def read_pdf(self, pdf_path: str):
        """读取pdf文件"""
        page_lists = []
        with pdfplumber.open(pdf_path) as pdf:
            for idx,page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                page_lists.append({
                    "page_num": idx,
                    "page_text": txt
                })
        text_splitter = RecursiveCharacterTextSplitter(
            
        )
        return page_lists
    
    def read_doc(self, doc_path: str):


