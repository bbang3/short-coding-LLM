import replicate
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
import jsonlines
import asyncio

"""
Dataset({
    features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'],
    num_rows: 164
})
"""

async def infer_async(dataset: Dataset, model: str):
    results = []

    # Run all prompts asynchronously
    prompts = dataset["prompt"]
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(replicate.async_run(model, input={"prompt": prompt}))
            for prompt in prompts
        ]

    compeletions = await asyncio.gather(*tasks)

    # Gather completions and format results
    for i, compeletion in enumerate(compeletions):
        completion = "".join([s for s in compeletion])
        results.append({
            "task_id": dataset["task_id"][i],
            "completion": completion
        })

    return results

def infer(dataset: Dataset, model: str):
    results = []

    # Run all prompts synchronously
    for item in tqdm(dataset):
        prompt = item["prompt"]

        output_replicate = replicate.run(
            model,
            input={
                "top_k": 50,
                "top_p": 1.0,
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0,
                "repeat_penalty": 1.1,
                "presence_penalty": 0,
                "frequency_penalty": 0
            }
        )
        
        # Replicate returns a list of strings synchronously
        completetion = "".join([s for s in output_replicate])
        results.append({
            "task_id": item["task_id"],
            "completion": completetion
        })
        break
        
    return results

def save_results(results):
    with jsonlines.open("results.jsonl", "w") as writer:
        writer.write_all(results)

async def main():
    dataset = load_dataset("openai_humaneval", split="test")
    results = await infer_async(dataset, "meta/codellama-34b-python:482ba325daab209d121f45a0030f2f3ed942df98b185d41635ab3f19165a3547")
    # results = infer(dataset, "meta/codellama-34b-python:482ba325daab209d121f45a0030f2f3ed942df98b185d41635ab3f19165a3547")
    save_results(results)


if __name__ == "__main__":
    asyncio.run(main())