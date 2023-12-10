import gzip
import json
import os
import re
from typing import Dict, Iterable

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

def data_loader(path:str = HUMAN_EVAL, mode:str = "problem")-> Dict[str, Dict]:
    """
    load Data from json, jsonl files
    Input Data Format:
        "problem": json/jsonl [task_id, prompt, entry_point, canonical_solution, test]
        "sample": json/jsonl [task_id, completion]
    
    Output Data Format:
        "problem": Dict {task_id: {prompt, func_name, solution, test}}
        "sample": Dict {task_id: {predict_code}}

    Parameters:
        path: path of data
        mode: "problem": return Dict of {task_id: {prompt, func_name, solution, test}}
              "sample": return List of {task_id: {predict_code}}
    """
    assert mode in ["problem", "sample"] 
    _json = json_parser(path)

    if mode == "problem":
        return {task["task_id"]: dict(prompt=task["prompt"], func_name=task["entry_point"], 
                                      solution=task["canonical_solution"],
                                      test=re.sub("\n*METADATA = *{[^}]*}\n*", "", task["test"])) for task in _json}
    elif mode == "sample":
        return {task["task_id"]:dict(predict_code=task["completion"]) for task in _json}
    

def data_writer(data:Dict, data_name:str = "temp_file"):
    '''
    write Dict data as json file
    '''
    with open(data_name+".json", "w") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent="\t")