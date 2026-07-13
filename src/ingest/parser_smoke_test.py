"""Run 3 PDF parsers over the same sample PDFs and dump their outputs
so we can eyeball which one best preserves text + tables + numbers."""
import time
from pathlib import Path

SAMPLES = [
    'APPLE_2022_10K',
    'Pfizer_2023Q2_10Q',
    'PEPSICO_2015_10K',
    'ULTABEAUTY_2023Q2_EARNINGS',
]

PDF_DIR = Path('data/raw_pdfs')
OUT_DIR = Path('data/parser_smoke')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_pypdf(path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return [(i + 1, (p.extract_text() or '')) for i, p in enumerate(reader.pages)]

def parse_pdfplumber(path):
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            pages.append((i + 1, page.extract_text() or ''))
    return pages

def parse_pymupdf(path):
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    pages = [(i + 1, doc.load_page(i).get_text('text')) for i in range(doc.page_count)]
    doc.close()
    return pages

PARSERS = {'pypdf': parse_pypdf, 'pdfplumber': parse_pdfplumber, 'pymupdf': parse_pymupdf}

print(f'{"doc":<34} {"parser":<12} {"pages":>6} {"chars":>10} {"secs":>7}')
print('-' * 74)

for doc in SAMPLES:
    src = PDF_DIR / f'{doc}.pdf'
    if not src.exists():
        print(f'{doc}: MISSING'); continue

    for pname, pfn in PARSERS.items():
        t0 = time.perf_counter()
        try:
            pages = pfn(src)
        except Exception as e:
            print(f'{doc:<34} {pname:<12} ERROR: {str(e)[:40]}')
            continue
        dt = time.perf_counter() - t0
        total_chars = sum(len(t) for _, t in pages)

        out = OUT_DIR / f'{pname}__{doc}.txt'
        with open(out, 'w', encoding='utf-8') as f:
            f.write(f'# parser={pname}  doc={doc}  pages={len(pages)}  chars={total_chars}  secs={dt:.2f}\n\n')
            for pnum, txt in pages:
                f.write(f'\n===== PAGE {pnum} =====\n{txt}\n')

        print(f'{doc:<34} {pname:<12} {len(pages):>6} {total_chars:>10} {dt:>7.2f}')
    print()
