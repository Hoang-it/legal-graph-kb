# Debug Phase 2 exit=1 (0s)

Thêm cell debug sau cell Phase 2 trong notebook để xem actual error:

## Cell 1: Đọc log files

```python
import os
for log in sorted(os.listdir('logs')):
    print(f'\n{"="*60}\n{log}\n{"="*60}')
    with open(f'logs/{log}', encoding='utf-8') as f:
        print(f.read())
```

## Cell 2: Test 1 cmd thủ công (xem error chính xác)

```python
!python -m experiments.run_inference --arm graphrag --n 1
```

## Causes phổ biến + fix

### 1. Env vars không propagate vào subprocess

**Symptom**: log có `KeyError: 'OPENAI_API_KEY'` hoặc `neo4j.exceptions.ServiceUnavailable`

**Fix**: re-run cell Phase 1.1 (secrets reload) — nếu user restart runtime mất env vars

### 2. cwd không đúng

**Symptom**: log có `No such file or directory: 'data/eval/questions_200.json'`

**Fix**: subprocess thừa kế cwd. Verify trong cell:
```python
import os; print(os.getcwd())
# Expected: /content/legal-graph-kb
```
Nếu không phải → re-run `%cd $REPO_DIR` trong Phase 1.2.

### 3. Module import error

**Symptom**: log có `ModuleNotFoundError: No module named 'experiments'`

**Fix**: subprocess cần `experiments/__init__.py`. Verify:
```python
!ls -la experiments/__init__.py src/__init__.py
```
Nếu missing → file đã được commit, chạy lại `git pull`.

### 4. Neo4j connection fail

**Symptom**: log có `neo4j.exceptions.ServiceUnavailable` hoặc connection refused

**Fix**: 
- Check Neo4j Aura instance vẫn running (free tier auto-pause sau 3 ngày inactivity)
- Re-test trong cell:
```python
from neo4j import GraphDatabase
import os
d = GraphDatabase.driver(os.environ['NEO4J_URI'],
    auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
with d.session(database='neo4j') as s:
    print(s.run('MATCH (n:Article) RETURN count(n)').single()[0])
```

### 5. SWI-Prolog missing (chỉ cho elite arms)

**Symptom**: log có `FileNotFoundError: 'swipl'`

**Fix**: 
```python
!which swipl || apt-get install -y swi-prolog
```

### 6. Path argument format

**Symptom**: log có argparse error về `--arm`

**Note**: `--arm` (singular) là deprecated nhưng vẫn support. `--arms` (plural) là format mới. Notebook dùng `--arm` cho R1 inference (5 arms riêng biệt) — đúng.
