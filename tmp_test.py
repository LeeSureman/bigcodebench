import argparse
from gradio_client import Client, handle_file
import jsonlines
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--')


    gradio_endpoint = "https://bigcode-bigcodebench-evaluator.hf.space/"
    client = Client(gradio_endpoint)
    results, pass_at_k = client.predict(
        split='complete',
        subset='hard',
        samples=handle_file('/home/xiaonan/mycode/bigcodebench/tmp_generation.jsonl'),
        pass_k="1,5,10",
        parallel=-1,
        min_time_limit=1,
        max_as_limit=30720,
        max_data_limit=30720,
        max_stack_limit=20,
        calibrated=True,
        check_gt_only=False,
        no_gt=False,
        selective_evaluate=None,
        api_name="/predict"
    )
    with jsonlines.open('test_results.jsonl', 'w') as f_out:
        f_out.write_all(results)
    print(results)
    print(pass_at_k)