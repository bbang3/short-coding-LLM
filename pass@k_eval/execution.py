from typing import Dict
import threading
import re

def calculate_code_length(code:str) ->int:
    def remove_comments(input_code):
        pattern = r'#.*|("""[\s\S]*?""")|(\'\'\'[\s\S]*?\'\'\')'
        code_without_comments = re.sub(pattern, '', input_code)
        return code_without_comments

    def remove_typing(code):
        codes = code.split("\n")
        code_without_typing = ""
        for code in codes:
            if "def" in code:
                _code = re.sub(r':\s*[^,\)]*', '', code)
                code = re.sub(r"->[^:]*:", ':', _code+":")
            code_without_typing+=code+"\n"
        return code_without_typing[:-1]
    
    comment_removed_code = re.sub(r"^[^\S\n]*\n","",remove_comments(code))
    typing_removed_code = remove_typing(comment_removed_code)
    return len(re.sub(r'[ \t]', '', typing_removed_code))

def execution_error_detect(_task_id:str, problem:Dict, predict_code:str, time_out:float)->Dict:
    '''
    Input Data Format:
        problem: Dict {prompt, func_name, test, ...}
    
    Output Data Format:
        Dict {task_id, passed, results}
        passed: True/False if all test cases have passed
        results: list of results in test cases
    '''

    def unsafe_excute(result, length):
        prompt = problem["prompt"]
        tests = problem["test"].split("\n")
        func_name = problem["func_name"]

        length.append(calculate_code_length(prompt +"\n" + predict_code ))
        for test in tests[1:]:
            if test == "":
                continue
            check_program = (
                prompt +"\n" +
                predict_code + "\n" +
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
    _length = []
    _thread = threading.Thread(target=unsafe_excute, args=(_result,_length,), daemon=True)
    _thread.start()

    _thread.join(timeout=time_out)

    if not _result:
        _result.append(0)

    _passed = 1 if all(r == 1 for r in _result) else 0

    return dict(task_id=_task_id, passed = _passed, result=_result, length=_length[0])