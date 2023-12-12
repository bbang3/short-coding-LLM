import evaluation
import os

if __name__=="__main__":
    in_dir = "predictions"
    out_dir = "evaluations"

    experiment_name = "summary_1"
    in_path = os.path.join(in_dir, f"prediction_{experiment_name}.jsonl")
    out_path = os.path.join(out_dir, f"evaluation_{experiment_name}")
    evaluation.evaluate(samplePath=in_path, to_json=True, json_name=out_path, add_error_list=True)