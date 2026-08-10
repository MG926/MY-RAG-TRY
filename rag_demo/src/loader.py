import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

""""该模块负责文件读取，目前仅支持pdf形式""""
class loader:
    def read_pdf(self, pdf_path: str):
        """读取pdf文件"""
        #设置分割器的块大小为500，重叠大小为80
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 500,
                chunk_overlap = 80
        )
        all_chunks = []
        #利用pdfplumber读取pdf文字内容，并且分chunk存储在all_chunks中
        with pdfplumber.open(pdf_path) as pdf:
            for idx,page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                chunks = text_splitter.split_text(txt)
                for chunk in chunks:
                    all_chunks.append({
                        "content": chunk,
                        "metadata": {"page_num": idx}
                    })
        return all_chunks



