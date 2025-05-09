from gradio_client import Client, handle_file


if __name__ == '__main__':
    gradio_endpoint = "https://bigcode-bigcodebench-evaluator.hf.space/"

    results, pass_at_k = client.predict(
        split=split,
        subset=subset,
        samples=handle_file(samples),
        pass_k=pass_k,
        parallel=parallel,
        min_time_limit=min_time_limit,
        max_as_limit=max_as_limit,
        max_data_limit=max_data_limit,
        max_stack_limit=max_stack_limit,
        calibrated=calibrated,
        check_gt_only=check_gt_only,
        no_gt=no_gt,
        selective_evaluate=selective_evaluate,
        api_name="/predict"
    )