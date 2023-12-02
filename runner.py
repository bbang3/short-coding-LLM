from typing import List
import json
import jsonlines
import argparse
import asyncio

import replicate 
from datasets import Dataset, load_dataset

class ReplicateRunner(object):
    def __init__(self, args):
        with open("config.json", "r") as f:
            config = json.load(f)
        self.params = config["params"]
        self.models = config["models"]
        
        self.results: List[dict] = []

    async def infer_baseline(self, dataset: Dataset):
        """
        Baseline: Naive generation of code
        """
        results = []

        # Run all prompts asynchronously
        prompts = dataset["prompt"]
        
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(replicate.async_run(
                    self.models["baseline"], 
                    input={
                        **self.params["baseline"],
                        "prompt": prompt
                    }
                    ))
                for prompt in prompts
            ]

        completions = await asyncio.gather(*tasks)
        completions = ["".join([s for s in completion]) for completion in completions]

        for i, completion in enumerate(completions):
            results.append({
                "task_id": dataset["task_id"][i],
                "completion": completion
            })

        self.results = results
        self.save_results(results)
        return results

    async def infer_summary(self, dataset: Dataset, summarize_prompt: str):
        """
        2-stage approach: generate code and then summarize it
        """
        results = await self.infer_baseline(dataset)
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(replicate.async_run(
                    self.models["summary"], 
                    input={
                        **self.params["summary"],
                        "prompt": summarize_prompt + "\n" + result["completion"]
                    }
                    ))
                for result in results
            ]

        self.results = results
        self.save_results(results)
        return results
    
    def save_results(self, results):
        with jsonlines.open("results.jsonl", "w") as writer:
            writer.write_all(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="baseline")
    args = parser.parse_args()

    runner = ReplicateRunner(args)
    dataset = load_dataset("openai_humaneval", split="test")

    if args.method == "summary":
        asyncio.run(runner.infer_summary(dataset, summarize_prompt="Summarize the following code:"))
    else:
        asyncio.run(runner.infer_baseline(dataset))