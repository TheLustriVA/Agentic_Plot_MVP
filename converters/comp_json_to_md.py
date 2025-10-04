import json
import argparse
from pathlib import Path
from datetime import datetime

def json_to_md(filepath:Path) -> list[str]:
    """Converts the JSON file at filepath to a list of Markdown-formatted strings."""

    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    test_collection = []

    for load in payload:
        test_result = []
        test_result.append(f"# Basic Export {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        test_result.append(f"Model: {load['model_name']}\n")
        test_result.append(f"Status: {load['status']}\n")
        test_result.append(f"Model Config: {load['model_config']}\n")
        test_result.append("\n## Creative Tests\n\n")
        for test in load['creative_tests']['test_results']:
            test_result.append(f"### {test['test_name']}\n\n")
            test_result.append("#### Prompt\n\n")
            test_result.append(f"{test['prompt']}\n\n")
            test_result.append("#### Response\n\n")
            test_result.append(f"{test['response']}\n\n")
            test_result.append(f"Word count: {test['word_count']}\n")
            test_result.append(f"Success: {test['success']}\n")
            test_result.append(f"Evaluation notes: {test['evaluation_notes']}\n")
        test_collection.append(test_result)
    
    return test_collection

def main():
    """CLI Interface for result conversion to markdown."""
    
    parser = argparse.ArgumentParser(description="Agentic Plot Creation - Test results converter")
    parser.add_argument("file", type=str, default="../comparison_results/detailed_results.json", help="Path to comparison results JSON File")
    
    args = parser.parse_args()
    
    input_file = Path(args.file)

    result_files = json_to_md(input_file)
    
    for result in result_files:
        markdown_file = Path(f"comparison_results/results_{datetime.now().strftime('%H%M%S')}.md")
        markdown_content = "".join(result)
        
        with open(markdown_file, "a", encoding="utf-8") as f:
            f.write(markdown_content)


if __name__ == "__main__":
    main()