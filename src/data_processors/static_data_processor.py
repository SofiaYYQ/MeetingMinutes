import os
from typing import List

from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

class StaticDataProcessor():
    @staticmethod
    def load_pdf_documents(folder_path:str)->List[Document]:
        pdf_loader = PDFReader(return_full_document=True)
        documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                filepath = os.path.join(folder_path, filename)
                docs = pdf_loader.load_data(file=filepath)
                documents.extend(docs)
        return documents

    @staticmethod
    def split_document_into_chunks(document: Document, chunk_size = 512, chunk_overlap = 0)->List[Document]:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
        )

        results = []
        chunks = splitter.split_text(document.text)
        for i in range(len(chunks)):
            document = Document(text=chunks[i], metadata=document.metadata)
            results.append(document)
        return results
    


