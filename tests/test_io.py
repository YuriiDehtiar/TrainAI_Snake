import json
from pathlib import Path
import tempfile
import threading
import unittest

from snake_v9.io import write_json


class AtomicPointerTests(unittest.TestCase):
    def test_reader_never_observes_a_partially_written_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            pointer = Path(directory) / 'latest.json'
            write_json(pointer, {'iteration': 0, 'checkpoint': 'initial'})
            failures = []

            def reader():
                for _ in range(300):
                    try:
                        row = json.loads(pointer.read_text(encoding='utf-8'))
                        if not {'iteration', 'checkpoint'} <= row.keys():
                            failures.append('incomplete pointer')
                    except Exception as error:
                        failures.append(str(error))

            thread = threading.Thread(target=reader)
            thread.start()
            for iteration in range(30):
                write_json(pointer, {'iteration': iteration, 'checkpoint': f'model_{iteration}'})
            thread.join()
            self.assertEqual(failures, [])


if __name__ == '__main__':
    unittest.main()
