from typing import Dict
import threading
import re

def execution_error_detect(_task_id:str, problem:Dict, predict_code:str, time_out:float)->Dict:
    '''
    Input Data Format:
        problem: Dict {prompt, func_name, test, ...}
    
    Output Data Format:
        Dict {task_id, passed, results}
        passed: True/False if all test cases have passed
        results: list of results in test cases
    '''

    def unsafe_excute(result):
        prompt = problem["prompt"]
        tests = problem["test"].split("\n")
        func_name = problem["func_name"]
        _pattern = re.compile('def.*?\n', re.DOTALL)
        _predict_code = re.sub(_pattern, "", predict_code)

        for test in tests[1:]:
            if test == "":
                continue
            check_program = (
                prompt +"\n" +
                _predict_code + "\n" +
                tests[0] + "\n" +
                test + "\n" +
                f"check({func_name})"
            )
        
            try:
                exec(check_program)
                result.append(1)
            except:
                result.append(0)

    _result = []

    _thread = threading.Thread(target=unsafe_excute, args=(_result,), daemon=True)
    _thread.start()

    _thread.join(timeout=time_out)

    if not _result:
        _result.append(0)

    _passed = 1 if all(r == 1 for r in _result) else 0

    return dict(task_id=_task_id, passed = _passed, result=_result)