"""Return ranked RAG passages and their exact source references for an investigation."""
import argparse
import json
from pathlib import Path

from tyremind.agents.knowledge import load_knowledge, search_passages
from tyremind.agents.supervisor import digest, export_json


def retrieve_for_investigation(query=None, limit=3, database=Path('data/tyremind.db'),
                               consumer_id='local-phase3', index_dir=Path('data/knowledge/chroma'),
                               root=Path('data/phase3')):
    if not 1 <= limit <= 10 or (query is not None and not query.strip()):
        raise ValueError('Supply a nonempty query and a limit between 1 and 10.')
    handoff = load_knowledge(database, consumer_id, index_dir, root)
    if handoff['action'] == 'NO_WORK':
        return handoff
    packet = json.loads(Path(handoff['output']).read_text(encoding='utf-8'))
    question = packet['query'] if query is None else query.strip()
    if question == packet['query'] and limit <= 3:
        matches = packet['passages'][:limit]
    else:
        snapshot = (packet['collection'], packet['snapshot_sha256'])
        matches = search_passages(index_dir, question, limit, snapshot)['passages']
    passages = []
    for rank, match in enumerate(matches, 1):
        source = match['source']
        reference = f"{source['title']}, PDF page {source['pdf_page']}"
        passages.append(dict(rank=rank, reference=reference, url=source['citation_url'], **match))
    result = dict(investigation_id=packet['investigation_id'], tyre_id=packet['tyre_id'],
                  collection=packet['collection'], snapshot_sha256=packet['snapshot_sha256'],
                  query=question, passages=passages)
    key = digest(json.dumps([question, limit, packet['snapshot_sha256']]))[:24]
    path = root/'retrieval'/packet['investigation_id']/f'{key}.json'
    result['output'] = str(path)
    export_json(path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query')
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--consumer-id', default='local-phase3')
    parser.add_argument('--index-dir', type=Path, default=Path('data/knowledge/chroma'))
    print(json.dumps(retrieve_for_investigation(**vars(parser.parse_args())), indent=2, ensure_ascii=True))
