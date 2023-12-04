import gzip
import json
import os
import re
from typing import Dict, Iterable, List

ROOT = os.getcwd()
HUMAN_EVAL = os.path.join(ROOT,"Resources", "HumanEval.jsonl.gz")
RESULT = os.path.join(ROOT, "Resources", "results.jsonl")

def json_parser(_file:str) -> Iterable[Dict]:
    
    if _file.endswith(".gz"):
        with open(_file, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(_file, "r", encoding="UTF8") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def data_loader(path:str = HUMAN_EVAL, mode:str = "problem")-> List[Dict]:
    """
    Parameters:
        path: path of data
        mode: problem = return List of {task_id, prompt, solution, test and entry_point}
              sample = return List of {task_id, completion}
    """
    assert mode in ["problem", "sample"] 
    _json = json_parser(path)
    data = []

    if mode == "problem":
        for task in _json:
            data.append(dict(task_id=task["task_id"], prompt=task["prompt"], func_name=task["entry_point"],
                            solution=task["canonical_solution"],test=re.sub("\n*METADATA = *{[^}]*}\n*", "", task["test"])))
    elif mode == "sample":
        for task in _json:
            data.append(dict(task_id=task["task_id"], completion=task["completion"]))
    return data