from glob import glob

from config.config import Settings
from data_extraction.pdf_culture_read import PDFProcessor

if __name__ == '__main__':

    settings = Settings()
    pdf_processor = PDFProcessor(settings.GROQ_API_KEY)

    FILE_PATH = 'src/data_extraction/files/'

    all_txts = glob(FILE_PATH + '*.txt')
    

    for txt in all_txts:
        pdf_processor.process_txt(txt)
