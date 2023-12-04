from typing import Dict
from tempfile import TemporaryDirectory
import os
import threading

def exction_error_detect(problem:Dict, predictCode:str, timeOut:float)->Dict:

    def unsafe_excute(result):
        with TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            check_program = (
                problem["prompt"]+"\n"+
                predictCode+"\n"+
                problem["test"]+"\n"+
                f"check({problem['func_name']})")
            
            try:
                os.chdir(tmp_dir)
                exec(check_program)
                result.append("passed")
            except TimeoutException:
                result.append("time out")
            except BaseException as e:
                result.append(f"failed: {e}")
            os.chdir(origin_cwd)

    origin_cwd = os.getcwd()
    
    result = []

    t = threading.Thread(target=unsafe_excute, args=(result, ))
    t.start()
    t.join(timeout=timeOut+1)

    if not result:
        result.append("time out")

    return dict(task_id=problem["task_id"], passed=result[0]=="passed", result=result[0])

class TimeoutException(Exception):
    ...