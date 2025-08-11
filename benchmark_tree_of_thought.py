# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark tree-of-thought reasoning workloads.

This script benchmarks the performance of tree-of-thought style reasoning
where multiple reasoning paths are generated and evaluated in parallel.

Example usage:
    python benchmark_tree_of_thought.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --num-problems 10 \
        --branches-per-step 3 \
        --reasoning-steps 4 \
        --output-len 256
"""

import argparse
import dataclasses
import json
import random
import time
from typing import Any, List, Optional

import numpy as np
from tqdm import tqdm

from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser


@dataclasses.dataclass
class TreeOfThoughtProblem:
    problem: str
    reasoning_prompt: str
    evaluation_prompt: str


def get_math_problems() -> List[TreeOfThoughtProblem]:
    """Generate mathematical reasoning problems for tree-of-thought."""
    problems = [
        TreeOfThoughtProblem(
            problem="Find the number of ways to arrange 5 people in a circle where 2 specific people cannot sit next to each other.",
            reasoning_prompt="Let me think step by step about this combinatorics problem:",
            evaluation_prompt="Evaluate the correctness of this reasoning approach:"
        ),
        TreeOfThoughtProblem(
            problem="A water tank can be filled by pipe A in 6 hours and by pipe B in 4 hours. If both pipes are opened together, how long will it take to fill the tank?",
            reasoning_prompt="I need to solve this step by step using rates:",
            evaluation_prompt="Check if this solution method is correct:"
        ),
        TreeOfThoughtProblem(
            problem="In a geometric sequence, the first term is 2 and the fourth term is 54. Find the sum of the first 6 terms.",
            reasoning_prompt="Let me work through this geometric sequence problem:",
            evaluation_prompt="Verify this geometric sequence solution:"
        ),
        TreeOfThoughtProblem(
            problem="A rectangular garden is 20m longer than it is wide. If the area is 300 square meters, find the dimensions.",
            reasoning_prompt="I'll set up equations to solve this area problem:",
            evaluation_prompt="Review this algebraic approach:"
        ),
        TreeOfThoughtProblem(
            problem="Find the minimum value of f(x) = x² + 4x + 7 and the x-coordinate where it occurs.",
            reasoning_prompt="I need to find the minimum of this quadratic function:",
            evaluation_prompt="Assess this optimization solution:"
        )
    ]
    return problems


def get_logic_problems() -> List[TreeOfThoughtProblem]:
    """Generate logical reasoning problems for tree-of-thought."""
    problems = [
        TreeOfThoughtProblem(
            problem="Five friends (Alice, Bob, Carol, Dave, Eve) are sitting in a row. Alice is not at either end. Bob is somewhere to the left of Carol. Dave is next to Eve. What are the possible seating arrangements?",
            reasoning_prompt="Let me systematically work through the constraints:",
            evaluation_prompt="Check if this logical deduction is sound:"
        ),
        TreeOfThoughtProblem(
            problem="In a village, every person either always tells the truth or always lies. You meet three people: A says 'B is a liar', B says 'C is a liar', C says 'A and B are both liars'. Who tells the truth?",
            reasoning_prompt="I need to analyze each person's statement for consistency:",
            evaluation_prompt="Verify this truth-teller/liar analysis:"
        ),
        TreeOfThoughtProblem(
            problem="A company has 100 employees. 60 speak English, 50 speak Spanish, 30 speak French. 20 speak both English and Spanish, 15 speak both English and French, 10 speak both Spanish and French. 5 speak all three languages. How many speak none of these languages?",
            reasoning_prompt="I'll use set theory and inclusion-exclusion principle:",
            evaluation_prompt="Review this set theory calculation:"
        )
    ]
    return problems


def create_tree_of_thought_prompts(
    problem: TreeOfThoughtProblem,
    branches_per_step: int,
    reasoning_steps: int
) -> List[str]:
    """Create prompts for tree-of-thought reasoning."""
    prompts = []
    
    # Initial reasoning branches
    for i in range(branches_per_step):
        prompt = f"""Problem: {problem.problem}

{problem.reasoning_prompt}

Approach {i+1}: Let me try a different reasoning path to solve this problem.
"""
        prompts.append(prompt)
    
    # Subsequent reasoning steps (simplified for benchmark)
    for step in range(1, reasoning_steps):
        for branch in range(branches_per_step):
            prompt = f"""Problem: {problem.problem}

Continuing from previous reasoning, let me explore step {step+1} of approach {branch+1}:
"""
            prompts.append(prompt)
    
    # Evaluation prompts
    for i in range(branches_per_step):
        eval_prompt = f"""Problem: {problem.problem}

{problem.evaluation_prompt}

Please evaluate approach {i+1} and rate its correctness and completeness on a scale of 1-10.
"""
        prompts.append(eval_prompt)
    
    return prompts


def run_tree_of_thought_benchmark(
    llm: LLM,
    problems: List[TreeOfThoughtProblem],
    branches_per_step: int,
    reasoning_steps: int,
    output_len: int,
    num_iterations: int
) -> List[float]:
    """Run the tree-of-thought benchmark."""
    
    sampling_params = SamplingParams(
        temperature=0.7,  # Some randomness for diverse reasoning paths
        top_p=0.9,
        max_tokens=output_len,
        n=1
    )
    
    latencies = []
    
    for iteration in tqdm(range(num_iterations), desc="Tree-of-thought iterations"):
        # Select a random problem
        problem = random.choice(problems)
        
        # Generate all prompts for this tree-of-thought instance
        prompts = create_tree_of_thought_prompts(
            problem, branches_per_step, reasoning_steps
        )
        
        # Measure latency for the entire tree-of-thought process
        start_time = time.perf_counter()
        
        # Generate responses for all reasoning branches and evaluation
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)
    
    return latencies


def save_results(args: argparse.Namespace, latencies: List[float]) -> None:
    """Save benchmark results to JSON files."""
    latencies_array = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies_array, percentages)
    
    results = {
        "benchmark_type": "tree_of_thought",
        "avg_latency": float(np.mean(latencies_array)),
        "latencies": latencies,
        "percentiles": dict(zip(percentages, percentiles.tolist())),
        "config": {
            "branches_per_step": args.branches_per_step,
            "reasoning_steps": args.reasoning_steps,
            "output_len": args.output_len,
            "num_problems": args.num_problems,
            "problem_type": args.problem_type
        }
    }
    
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        
        # Convert to PyTorch benchmark format
        pt_records = convert_to_pytorch_benchmark_format(
            args=args,
            metrics={"latency": latencies},
            extra_info={k: results[k] for k in ["avg_latency", "percentiles", "config"]},
        )
        if pt_records:
            pt_file = f"{args.output_json.rsplit('.', 1)[0]}.pytorch.json"
            write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    print("Tree-of-Thought Reasoning Benchmark")
    print(f"Configuration: {args}")
    
    # Initialize the LLM
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    
    # Get problems based on type
    if args.problem_type == "math":
        all_problems = get_math_problems()
    elif args.problem_type == "logic":
        all_problems = get_logic_problems()
    else:  # mixed
        all_problems = get_math_problems() + get_logic_problems()
    
    # Select subset of problems
    problems = random.sample(all_problems, min(args.num_problems, len(all_problems)))
    
    print(f"Selected {len(problems)} {args.problem_type} problems")
    print(f"Each problem will generate {args.branches_per_step * args.reasoning_steps + args.branches_per_step} prompts")
    
    # Warmup
    print("Warming up...")
    warmup_problem = problems[0]
    warmup_prompts = create_tree_of_thought_prompts(
        warmup_problem, args.branches_per_step, args.reasoning_steps
    )
    warmup_sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.output_len,
        n=1
    )
    
    for _ in range(args.num_warmup):
        llm.generate(warmup_prompts, sampling_params=warmup_sampling_params, use_tqdm=False)
    
    # Run benchmark
    print("Running tree-of-thought benchmark...")
    latencies = run_tree_of_thought_benchmark(
        llm=llm,
        problems=problems,
        branches_per_step=args.branches_per_step,
        reasoning_steps=args.reasoning_steps,
        output_len=args.output_len,
        num_iterations=args.num_iterations
    )
    
    # Print results
    latencies_array = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies_array, percentages)
    
    print(f"\nResults:")
    print(f"Average latency: {np.mean(latencies_array):.3f} seconds")
    print(f"Total prompts per iteration: {args.branches_per_step * args.reasoning_steps + args.branches_per_step}")
    print(f"Average latency per prompt: {np.mean(latencies_array) / (args.branches_per_step * args.reasoning_steps + args.branches_per_step):.3f} seconds")
    
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile:.3f} seconds")
    
    # Save results
    save_results(args, latencies)


def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark tree-of-thought reasoning workloads"
    )
    
    # Tree-of-thought specific arguments
    parser.add_argument(
        "--num-problems",
        type=int,
        default=5,
        help="Number of different problems to use in benchmark"
    )
    parser.add_argument(
        "--branches-per-step",
        type=int,
        default=3,
        help="Number of reasoning branches to explore per step"
    )
    parser.add_argument(
        "--reasoning-steps",
        type=int,
        default=3,
        help="Number of reasoning steps in the tree"
    )
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=["math", "logic", "mixed"],
        default="mixed",
        help="Type of problems to use for reasoning"
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per reasoning step"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Number of tree-of-thought iterations to run"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save results in JSON format"
    )
    
    # Add engine arguments
    parser = EngineArgs.add_cli_args(parser)
    
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)