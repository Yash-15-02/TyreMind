"""Extract page text and produce source-preserving, page-bounded RAG passages."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text or '').replace('\x00', '')
    return '\n'.join(re.sub(r'[\t ]+', ' ', line).strip() for line in text.splitlines()).strip()


def split_page(text, max_words=180, overlap=30):
    """Yield exact character slices of normalized page text, with word overlap."""
    if max_words < 10 or not 0 <= overlap < max_words:
        raise ValueError('Use max_words >= 10 and 0 <= overlap < max_words.')
    words = list(re.finditer(r'\S+', text))
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        if end < len(words):
            # Prefer a nearby sentence boundary without exceeding the word budget.
            for candidate in range(end, max(start + overlap + 1, end - 30), -1):
                if re.search(r'[.!?]["\u201d\u2019)]*$', words[candidate - 1].group()):
                    end = candidate
                    break
        char_start, char_end = words[start].start(), words[end - 1].end()
        yield {'text': text[char_start:char_end], 'char_start': char_start,
               'char_end': char_end, 'word_count': end - start}
        if end == len(words):
            break
        start = end - overlap


def chunk_documents(catalog=Path('knowledge/sources.json'), raw=Path('data/knowledge/raw'),
                    output=Path('data/knowledge/processed'), max_words=180, overlap=30):
    if max_words < 10 or not 0 <= overlap < max_words:
        raise ValueError('Use max_words >= 10 and 0 <= overlap < max_words.')
    raw, output = Path(raw), Path(output)
    if raw.resolve() == output.resolve() or raw.resolve() in output.resolve().parents:
        raise ValueError('Processed output must be separate from raw documents.')
    sources = json.loads(Path(catalog).read_text(encoding='utf-8'))['sources']
    pages, chunks, summaries = [], [], []
    for source in sources:
        if not source.get('enabled', True):
            summaries.append({'document_id': source['id'], 'status': 'disabled', 'chunks': 0})
            continue
        filename = source['filename']
        if Path(filename).name != filename:
            raise ValueError('Invalid source filename.')
        path = raw / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source.get('verified_sha256'):
            raise ValueError(f"{source['id']}: PDF differs from verified source; review page policy before chunking.")
        reader = PdfReader(path)
        if len(reader.pages) != source['expected_pages']:
            raise ValueError(f"{source['id']}: unexpected page count.")
        sections = source.get('chunk_sections', [])
        for section in sections:
            first, last = section['pdf_pages']
            if not 1 <= first <= last <= len(reader.pages):
                raise ValueError('Invalid section page range.')
        count, selected_pages, empty_pages = 0, 0, []
        for number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text())
            matching = [section for section in sections if section['pdf_pages'][0] <= number <= section['pdf_pages'][1]]
            if len(matching) > 1:
                raise ValueError('Overlapping section policy.')
            section = matching[0] if matching else None
            metadata = {'document_id': source['id'], 'title': source['title'],
                        'publisher': source['publisher'], 'source_url': source['url'],
                        'landing_url': source['landing_url'], 'edition': source['edition'],
                        'region': source['region'], 'filename': filename, 'source_sha256': digest,
                        'pdf_page': number, 'citation_url': f"{source['url']}#page={number}",
                        'section': section['section'] if section else 'Outside selected retrieval sections',
                        'printed_page': str(number + section['printed_page_offset'])
                        if section and 'printed_page_offset' in section else None}
            pages.append({**metadata, 'text': text, 'selected_for_chunking': section is not None})
            if not text:
                empty_pages.append(number)
                continue
            if section is None:
                continue
            selected_pages += 1
            for index, fragment in enumerate(split_page(text, max_words, overlap), start=1):
                identity = f"{source['id']}:{digest}:{number}:{fragment['char_start']}:{fragment['char_end']}"
                chunks.append({**metadata, **fragment,
                               'chunk_id': hashlib.sha256(identity.encode()).hexdigest(),
                               'page_chunk_number': index, 'extraction_method': 'pypdf',
                               'text_sha256': hashlib.sha256(fragment['text'].encode()).hexdigest()})
                count += 1
        summaries.append({'document_id': source['id'], 'status': 'extracted',
                          'pages': len(reader.pages), 'selected_pages': selected_pages,
                          'empty_pages': empty_pages, 'chunks': count,
                          'exclusion_note': source.get('chunk_exclusion_reason')})
    if not chunks:
        raise ValueError('No passages produced; inspect extraction and section policy.')
    report = {'chunk_version': '1.0', 'max_words': max_words, 'overlap_words': overlap,
              'page_count': len(pages), 'chunk_count': len(chunks), 'documents': summaries,
              'offset_reference': 'char_start inclusive / char_end exclusive in normalized pages.jsonl text',
              'notice': 'Page-bounded extraction can lose diagram/table layout and cross-page context. No embeddings built.'}
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in [('pages.jsonl', pages), ('chunks.jsonl', chunks)]:
        temporary = output / (filename + '.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        temporary.replace(output / filename)
    (output / 'chunk_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('knowledge/sources.json'))
    parser.add_argument('--input', type=Path, default=Path('data/knowledge/raw'))
    parser.add_argument('--output', type=Path, default=Path('data/knowledge/processed'))
    parser.add_argument('--max-words', type=int, default=180)
    parser.add_argument('--overlap', type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(chunk_documents(args.catalog, args.input, args.output, args.max_words, args.overlap), indent=2))


if __name__ == '__main__':
    main()
