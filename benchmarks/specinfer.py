import flexflow.serve as ff
import argparse
import json
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--memory_per_gpu', type=int, default=48500)
    parser.add_argument('--zero_copy_memory_per_node', type=int, default=30000)
    parser.add_argument('--tensor_parallelism_degree', type=int, default=1)
    parser.add_argument('--pipeline_parallelism_degree', type=int, default=1)
    parser.add_argument('--llm', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--ssm', type=str, default="JackFram/llama-68m")
    parser.add_argument('--max_requests_per_batch', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--max_tokens_per_batch', type=int, default=128) 
    args = parser.parse_args()

    dataset = json.load(open("chatgpt.json"))

    ff.init(num_gpus=args.num_gpus,
            memory_per_gpu=args.memory_per_gpu,
            zero_copy_memory_per_node=args.zero_copy_memory_per_node,
            tensor_parallelism_degree=args.tensor_parallelism_degree,
            pipeline_parallelism_degree=args.pipeline_parallelism_degree
        )
    # Specify the LLM
    llm = ff.LLM(args.llm)

    # Specify a list of SSMs (just one in this case)
    ssms=[]
    if args.ssm != '':
        ssm_names = args.ssm.split(',')
        for ssm_name in ssm_names:
            ssm = ff.SSM(ssm_name)
            ssms.append(ssm)

    # Create the sampling configs
    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0, topp=1, topk=1
    )

    # Compile the SSMs for inference and load the weights into memory
    for ssm in ssms:
        ssm.compile(generation_config,
                    max_requests_per_batch=args.max_requests_per_batch,
                    max_seq_length=args.max_seq_length,
                    max_tokens_per_batch=args.max_tokens_per_batch)

    # Compile the LLM for inference and load the weights into memory
    llm.compile(generation_config, 
                ssms=ssms,
                max_requests_per_batch=args.max_requests_per_batch,
                max_seq_length=args.max_seq_length,
                max_tokens_per_batch=args.max_tokens_per_batch
               )

    llm.start_server()

    start_time = time.perf_counter()
    result = llm.generate(requests_or_prompts=dataset[:16])
    end_time = time.perf_counter()

    print(result)
    print(f"request throughput: {len(dataset) / (end_time - start_time)} requests/second")