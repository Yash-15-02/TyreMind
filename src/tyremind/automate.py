"""Run monitoring, investigation delivery and the fleet dashboard together."""
import argparse
from contextlib import closing
import logging
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from tyremind.agents.investigation_loop import run_loop
from tyremind.agents.phase4_handoff import publish
from tyremind.agents.prepare_outcome import prepare_outcome
from tyremind.agents.save_outcome import save_outcome
from tyremind.agents.validate_outcome import validate_outcome


def pending_delivery(database):
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'investigations' not in tables:
            return None
        unseen = ('AND NOT EXISTS (SELECT 1 FROM phase4_inbox p WHERE '
                  'p.investigation_id=i.investigation_id)' if 'phase4_inbox' in tables else '')
        row = db.execute("""SELECT i.investigation_id FROM investigations i
            WHERE status IN ('COMPLETED','FAILED')
            AND json_extract(result_json,'$.workflow')='agent_investigation' """
            + unseen + ' ORDER BY finished_at,investigation_id LIMIT 1').fetchone()
        return row[0] if row else None


def process_next(database=Path('data/tyremind.db'), phase3=Path('data/phase3'),
                 local_model='llama3.2:3b'):
    identity = pending_delivery(database)
    if identity is None:
        progress = run_loop(database, phase3, local_model=local_model)
        if progress['stage'] == 'NO_WORK':
            return {'action': 'NO_WORK'}
        identity = progress['investigation_id']
        prepare_outcome(database, phase3, local_model=local_model)
        validate_outcome(database, phase3, local_model=local_model)
        save_outcome(identity, database)
    return publish(identity, database, phase3.parent/'phase4'/'inbox')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-ollama', default='llama3.2:3b')
    args = parser.parse_args()
    if not Path('data/tyremind.db').is_file():
        parser.error('Run from the prepared project folder containing data/tyremind.db.')
    logging.basicConfig(level=logging.INFO)
    dashboard = Path(__file__).parent/'phase4'/'dashboard.py'
    children = []
    try:
        for module_args in (['tyremind.monitoring.worker'], ['streamlit', 'run', str(dashboard)]):
            children.append(subprocess.Popen([sys.executable, '-m', *module_args]))
        while all(child.poll() is None for child in children):
            try:
                result = process_next(local_model=args.local_ollama)
            except Exception:
                logging.exception('Investigation paused; retrying the saved checkpoint in 30 seconds.')
                time.sleep(30)
                continue
            if result['action'] == 'NO_WORK':
                time.sleep(5)
            else:
                logging.info('Investigation delivery: %s', result)
        raise SystemExit('Monitoring or dashboard exited; see its output above.')
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()


if __name__ == '__main__':
    main()
