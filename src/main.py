from glob import glob

from config.config import Settings
from data_extraction.pdf_culture_read import PDFProcessor

if __name__ == '__main__':

    settings = Settings()
    pdf_processor = PDFProcessor(settings.GROQ_API_KEY)

    FILE_PATH = 'src/data_extraction/files/'

    all_pdfs = glob(FILE_PATH + '*.pdf')

    for pdf in all_pdfs:
        pdf_processor.process_pdf(pdf)
