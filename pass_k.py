from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List
import numpy as np
import itertools
from collections import defaultdict
import tqdm

import utils
from execution_thread import exction_error_detect

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
        timeOut:float = 3.0,
        nWorkers:int = 1
):
    problems = utils.data_loader(path=problemPath, mode="problem")
    samples = utils.data_loader(path=samplePath, mode="sample")
    assert len(problems)==len(samples)

    with ThreadPoolExecutor(max_workers=nWorkers) as executor:
        
        preds = list()
        results = defaultdict(list)

        print("Finding Error in Samples...")
        for i in tqdm.tqdm(range(len(samples))):
            args = (problems[i], samples[i]["completion"], timeOut)
            pred = executor.submit(exction_error_detect, *args)
            preds.append(pred)

        print("Running test suites...")
        for pred in tqdm.tqdm(as_completed(preds)):
            result = pred.result()
            results[result["task_id"]].append((result))

        total, correct = list(), list()
        for result in results.values():
            result.sort()
            passed = [r["passed"] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        if isinstance(k, int):
            pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct,k)}
        else:
            ks = k
            pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct,k).mean() for k in ks if (total>=k).all()}

        return pass_at_k