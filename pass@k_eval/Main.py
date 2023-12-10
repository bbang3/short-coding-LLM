import evaluation

if __name__=="__main__":
    # evaluation.evaluate(to_json=True, add_error_list=True)
    evaluation.evaluate(samplePath="predictions\predictions_instruct_baseline.jsonl", to_json=True, json_name="i", add_error_list=True)