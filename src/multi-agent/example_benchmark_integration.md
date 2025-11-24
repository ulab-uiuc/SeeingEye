python3
```
#!/usr/bin/env python3
"""
Example: Integrating FlowExecutor with Other Benchmarks

This example shows how to use the modular FlowExecutor with different benchmark formats.
The FlowExecutor is benchmark-agnostic and can be easily adapted to any evaluation framework.
"""

import json
from datetime import datetime
from app.flow.flow_executor import FlowExecutor


class GenericBenchmarkAdapter:
    """Example adapter for any benchmark format."""

    def __init__(self, max_iterations=3):
        self.flow_executor = FlowExecutor(max_iterations=max_iterations)

    def run_evaluation(self, questions: list) -> list:
        """Run evaluation on a list of questions."""
        results = []

        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}")

            # Execute flow
            result = self.flow_executor.execute(question)

            results.append({
                "question_id": i + 1,
                "question": question,
                "response": result["response"],
                "execution_time": result["execution_time_seconds"],
                "success": result["success"]
            })

        return results


class VQABenchmarkAdapter:
    """Example adapter for VQA-style benchmarks."""

    def __init__(self, max_iterations=3):
        self.flow_executor = FlowExecutor(max_iterations=max_iterations)

    def format_vqa_input(self, question: str, image_path: str = None) -> str:
        """Format VQA input for the flow."""
        if image_path:
            return f"{question}\nimage_path:{image_path}"
        return question

    def run_vqa_evaluation(self, vqa_samples: list) -> list:
        """Run evaluation on VQA format samples."""
        results = []

        for sample in vqa_samples:
            question = sample.get("question", "")
            image_path = sample.get("image_path")
            expected_answer = sample.get("answer")

            # Format input for flow
            flow_input = self.format_vqa_input(question, image_path)

            # Execute flow
            result = self.flow_executor.execute(flow_input)

            results.append({
                "question_id": sample.get("id"),
                "question": question,
                "expected_answer": expected_answer,
                "response": result["response"],
                "execution_time": result["execution_time_seconds"],
                "success": result["success"]
            })

        return results


class MathBenchmarkAdapter:
    """Example adapter for math reasoning benchmarks."""

    def __init__(self, max_iterations=3):
        self.flow_executor = FlowExecutor(max_iterations=max_iterations)

    def format_math_input(self, problem: str, choices: list = None) -> str:
        """Format math problem for the flow."""
        if choices:
            choices_str = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            return f"{problem}\n\nOptions:\n{choices_str}"
        return problem

    def run_math_evaluation(self, math_problems: list) -> list:
        """Run evaluation on math problems."""
        results = []

        for problem in math_problems:
            question = problem.get("problem", "")
            choices = problem.get("choices")
            expected_answer = problem.get("answer")

            # Format input for flow
            flow_input = self.format_math_input(question, choices)

            # Execute flow
            result = self.flow_executor.execute(flow_input)

            results.append({
                "problem_id": problem.get("id"),
                "problem": question,
                "expected_answer": expected_answer,
                "response": result["response"],
                "execution_time": result["execution_time_seconds"],
                "success": result["success"]
            })

        return results


def example_usage():
    """Demonstrate how to use the modular flow with different benchmarks."""

    # Example 1: Generic text questions
    print("🧪 Example 1: Generic Benchmark")
    generic_adapter = GenericBenchmarkAdapter(max_iterations=2)

    questions = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "Solve: 2x + 5 = 15"
    ]

    results = generic_adapter.run_evaluation(questions)
    print(f"Processed {len(results)} questions")
    print("-" * 50)

    # Example 2: VQA-style benchmark
    print("🖼️ Example 2: VQA Benchmark")
    vqa_adapter = VQABenchmarkAdapter(max_iterations=2)

    vqa_samples = [
        {
            "id": 1,
            "question": "What objects are visible in this image?",
            "image_path": "/path/to/image1.jpg",
            "answer": "car, tree, building"
        },
        {
            "id": 2,
            "question": "What is the weather like?",
            "image_path": "/path/to/image2.jpg",
            "answer": "sunny"
        }
    ]

    vqa_results = vqa_adapter.run_vqa_evaluation(vqa_samples)
    print(f"Processed {len(vqa_results)} VQA samples")
    print("-" * 50)

    # Example 3: Math reasoning benchmark
    print("🔢 Example 3: Math Benchmark")
    math_adapter = MathBenchmarkAdapter(max_iterations=2)

    math_problems = [
        {
            "id": 1,
            "problem": "If a train travels 60 miles in 2 hours, what is its speed?",
            "choices": ["20 mph", "30 mph", "40 mph", "60 mph"],
            "answer": "B"
        },
        {
            "id": 2,
            "problem": "What is the derivative of x^2?",
            "answer": "2x"
        }
    ]

    math_results = math_adapter.run_math_evaluation(math_problems)
    print(f"Processed {len(math_results)} math problems")
    print("-" * 50)

    # Save all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "generic_benchmark": results,
        "vqa_benchmark": vqa_results,
        "math_benchmark": math_results
    }

    with open("example_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("✅ All results saved to example_benchmark_results.json")


if __name__ == "__main__":
    example_usage()
```