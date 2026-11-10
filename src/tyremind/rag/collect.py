"""Download cataloged original PDFs with source metadata and checksums."""
import argparse
import gzip
import io
import hashlib
import json
import os
import ssl
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

MAX_BYTES = 25 * 1024 * 1024


def collect(catalog=Path('knowledge/sources.json'), output=Path('data/knowledge/raw')):
    entries = [source for source in json.loads(Path(catalog).read_text(encoding='utf-8'))['sources']
               if source.get('enabled', True)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Respect managed environments' CA settings; never disable TLS verification.
    cafile = os.environ.get('SSL_CERT_FILE') or os.environ.get('REQUESTS_CA_BUNDLE')
    context = ssl.create_default_context(cafile=cafile)
    results = []
    for source in entries:
        filename = source['filename']
        if Path(filename).name != filename:
            raise ValueError('Catalog filenames must not contain directory components.')
        if not source['url'].startswith('https://'):
            raise ValueError('Only HTTPS source URLs are supported.')
        record = {**source, 'status': 'failed'}
        temporary = None
        try:
            request = Request(source['url'], headers={'User-Agent': 'TyreMind-Research/0.1'})
            with urlopen(request, timeout=45, context=context) as response:
                if not response.geturl().startswith('https://'):
                    raise ValueError('Source redirected away from HTTPS.')
                payload = response.read(MAX_BYTES + 1)
                if len(payload) > MAX_BYTES:
                    raise ValueError('Document exceeds the 25 MiB collection limit.')
                if payload.startswith(b'\x1f\x8b'):
                    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
                        payload = compressed.read(MAX_BYTES + 1)
                    if len(payload) > MAX_BYTES:
                        raise ValueError('Decompressed document exceeds 25 MiB.')
                if not payload.startswith(b'%PDF-') or b'%%EOF' not in payload[-4096:]:
                    raise ValueError('Response is not a complete PDF; download rejected.')
                record.update(resolved_url=response.geturl(), content_type=response.headers.get('Content-Type'))
            with tempfile.NamedTemporaryFile(dir=output, suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            temporary.replace(output / filename)
            record.update(status='downloaded', byte_count=len(payload),
                          sha256=hashlib.sha256(payload).hexdigest(),
                          downloaded_at=datetime.now(timezone.utc).isoformat())
        except Exception as error:
            record['error'] = str(error)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        results.append(record)
        print(f"{source['id']}: {record['status']}", flush=True)
    manifest = {'documents': results, 'downloaded': sum(r['status'] == 'downloaded' for r in results),
                'failed': sum(r['status'] != 'downloaded' for r in results)}
    (output / 'collection_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('knowledge/sources.json'))
    parser.add_argument('--output', type=Path, default=Path('data/knowledge/raw'))
    args = parser.parse_args()
    result = collect(args.catalog, args.output)
    print(f"Downloaded {result['downloaded']}; failed {result['failed']}.")
    if result['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
