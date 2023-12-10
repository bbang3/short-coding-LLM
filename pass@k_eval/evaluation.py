from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Tuple, Dict
import numpy as np
import itertools
from collections import defaultdict
import tqdm

import utils
from execution import execution_error_detect

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def evaluate(
        samplePath:str = utils.RESULT,
        problemPath:str = utils.HUMAN_EVAL,
        k:Union[int, List[int]] = [1,10,100],
        timeOut:float = 1.0,
        to_json:bool = False,
        add_error_list:bool = False
)->Tuple[Dict, List]:

    problems = utils.data_loader(path=problemPath, mode="problem")
    samples = utils.data_loader(path=samplePath, mode="sample")
    assert len(problems)==len(samples)

    with ThreadPoolExecutor(max_workers=1) as executor:
        
        preds = list()
        results = defaultdict(list)
        errors = list()
        pass_at_k = defaultdict(float)

        print("Finding Error in Samples...")
        for task_id in tqdm.tqdm(problems.keys()):
            args = (task_id, problems[task_id], samples[task_id]["predict_code"], timeOut)
            pred = executor.submit(execution_error_detect, *args)
            preds.append(pred)

        print("Running test suites...")
        for pred in as_completed(preds):
            result = pred.result()
            if not result["passed"]:
                if add_error_list:
                    errors.append((result["task_id"], result["result"]))
                else:
                    errors.append(result["task_id"])  
            results[result["task_id"]].append(result["result"])

        print("Calculating the results...")
        total, correct = list(), list()
        for passed in results.values():
            passed = passed[0]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        def _pass_at_k(_k):
            indices = np.where(total >= _k)[0]
            _total = total[indices]
            _correct = correct[indices]
            if _total.size==0 and _correct.size==0:
                return f"number of all test cases are less then {_k}"
            return estimate_pass_at_k(_total,_correct,_k).mean()
            
        if isinstance(k, int):
            pass_at_k = {f"pass@{k}":_pass_at_k(k)}
        else:
            ks = k
            for k in ks:
                pass_at_k[f"pass@{k}"] = _pass_at_k(k)
        print("Complete calculating")

        if to_json:
            pass_at_k["error_ids"] = errors
            utils.data_writer(pass_at_k, "eval_result")

        return pass_at_k, errors