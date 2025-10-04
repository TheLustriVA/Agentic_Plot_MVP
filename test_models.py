#!/usr/bin/env python3
"""
Single Model Testing Script for Agentic Plot MVP

This script tests each model individually for creative writing capabilities,
optimized for RTX 5090's 32GB VRAM constraint.
"""

import sys
import argparse
import json
from pathlib import Path
from agentic_plot.single_model_agent import SingleModelAgent


def create_creative_writing_tests() -> list:
    """Create standardized creative writing test scenarios"""
    return [
        {
            "name": "Character Development",
            "prompt": """Create a detailed character description for a protagonist in a cyberpunk noir story. 
            The character should be a former corporate security specialist turned private investigator. 
            Include physical appearance, personality, background, skills, and a distinctive quirk or flaw. 
            Write 300-400 words.""",
            "max_tokens": 600,
            "temperature": 0.7,
            "evaluation_notes": "Tests character creation, world-building integration, psychological depth"
        },
        {
            "name": "Dialogue Writing",
            "prompt": """Write a tense dialogue scene between two old friends who now find themselves on opposite sides of a conflict. 
            One is a rebel leader, the other works for the authoritarian government they're fighting. 
            They meet secretly in an abandoned subway station. 
            Include subtext, emotional tension, and natural conversation flow. 200-300 words.""",
            "max_tokens": 500,
            "temperature": 0.8,
            "evaluation_notes": "Tests dialogue quality, character voice, emotional depth, subtext"
        },
        {
            "name": "Scene Description",
            "prompt": """Describe a bustling alien marketplace on a distant planet. 
            Include unique alien species, strange foods and goods, unusual architecture, 
            and sensory details (sounds, smells, textures). 
            Make it feel alive and immersive while maintaining internal consistency. 250-350 words.""",
            "max_tokens": 500,
            "temperature": 0.7,
            "evaluation_notes": "Tests descriptive writing, world-building, creativity, sensory detail"
        },
        {
            "name": "Plot Synopsis",
            "prompt": """Write a compelling synopsis for a mystery novel set in Victorian London. 
            The story involves a series of impossible murders where the victims are found in locked rooms 
            with no apparent way for the killer to enter or escape. 
            Include the detective, key suspects, major plot points, and resolution. 400-500 words.""",
            "max_tokens": 700,
            "temperature": 0.6,
            "evaluation_notes": "Tests plot structure, mystery logic, pacing, narrative coherence"
        },
        {
            "name": "Creative Problem Solving",
            "prompt": """A spaceship crew is stranded on an ice planet with failing life support. 
            They have limited resources and conflicting personalities. 
            Describe how they work together (or fail to) to solve their predicament. 
            Focus on character interactions, creative solutions, and emotional stakes. 350-450 words.""",
            "max_tokens": 650,
            "temperature": 0.8,
            "evaluation_notes": "Tests creative thinking, character dynamics, tension building, problem resolution"
        }
    ]


def run_single_model_test(model_name: str, agent: SingleModelAgent, input_file: str = None) -> dict:
    """Run comprehensive test on a single model"""
    
    print(f"\n{'='*60}")
    print(f"🧪 TESTING MODEL: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Start the model
    if not agent.start_single_model(model_name):
        return {"model_name": model_name, "status": "failed_to_start", "results": None}
    
    try:
        # Get model info
        config = agent.MODELS[model_name]
        print("\n📋 Model Information:")
        print(f"   Description: {config.description}")
        print(f"   VRAM Estimate: {config.vram_estimate}GB")
        print(f"   Capabilities: {', '.join(config.capabilities)}")
        
        results = {
            "model_name": model_name,
            "status": "completed",
            "model_config": config,
            "creative_tests": None,
            "story_generation": None
        }
        
        # Run creative writing tests
        print("\n🎨 Running Creative Writing Tests...")
        test_scenarios = create_creative_writing_tests()
        creative_results = agent.test_creative_writing(test_scenarios)
        results["creative_tests"] = creative_results
        
        # Calculate success rate
        successful_tests = sum(1 for test in creative_results["test_results"] if test["success"])
        success_rate = (successful_tests / len(test_scenarios)) * 100
        print(f"\n📊 Creative Tests Success Rate: {success_rate:.1f}% ({successful_tests}/{len(test_scenarios)})")
        
        # Run story generation if input file provided
        if input_file and Path(input_file).exists():
            print("\n📚 Generating Complete Story Output...")
            story_outputs = agent.generate_full_story_output(input_file, f"outputs_{model_name}")
            results["story_generation"] = story_outputs
            
            print("✅ Story generation completed!")
            print(f"   Generated {len(story_outputs)} files in outputs_{model_name}/")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return {"model_name": model_name, "status": "error", "error": str(e), "results": None}
    
    finally:
        # Always stop the server
        agent.stop_server()
        print(f"🛑 Stopped {model_name} server")


def generate_comparison_report(all_results: list, output_dir: str = "comparison_results"):
    """Generate a comprehensive comparison report"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate summary report
    report = f"""# Agentic Plot MVP - Model Comparison Report

## Test Overview

**Date**: {Path().ctime()}
**VRAM Constraint**: 32GB (RTX 5090)
**Models Tested**: {len(all_results)}

## Model Performance Summary

"""
    
    for result in all_results:
        model_name = result["model_name"]
        status = result["status"]
        
        report += f"### {model_name.upper()}\n\n"
        
        if status == "completed":
            config = result.get("model_config")
            if config:
                report += f"- **VRAM**: {config.vram_estimate}GB\n"
                report += f"- **Context**: {config.context_size}\n"
                report += f"- **Description**: {config.description}\n"
            
            # Creative test results
            creative_tests = result.get("creative_tests", {})
            if creative_tests and "test_results" in creative_tests:
                successful = sum(1 for test in creative_tests["test_results"] if test["success"])
                total = len(creative_tests["test_results"])
                success_rate = (successful / total) * 100 if total > 0 else 0
                
                report += f"- **Creative Tests**: {successful}/{total} ({success_rate:.1f}%)\n"
                
                # Average word count
                word_counts = [test["word_count"] for test in creative_tests["test_results"] if test["success"]]
                avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
                report += f"- **Avg Output Length**: {avg_words:.0f} words\n"
            
            # Story generation
            story_gen = result.get("story_generation")
            if story_gen:
                report += f"- **Story Files Generated**: {len(story_gen)}\n"
            
            report += "- **Status**: ✅ Completed\n\n"
            
        elif status == "failed_to_start":
            report += "- **Status**: ❌ Failed to start\n\n"
        else:
            error = result.get("error", "Unknown error")
            report += f"- **Status**: ❌ Error - {error}\n\n"
    
    # Add detailed test results
    report += "\n## Detailed Test Results\n\n"
    
    for result in all_results:
        if result["status"] == "completed" and "creative_tests" in result:
            model_name = result["model_name"]
            report += f"### {model_name.upper()} - Creative Writing Tests\n\n"
            
            for test in result["creative_tests"]["test_results"]:
                status_icon = "✅" if test["success"] else "❌"
                report += f"**{test['test_name']}** {status_icon}\n"
                if test["success"]:
                    report += f"- Word count: {test['word_count']}\n"
                    report += f"- Evaluation: {test['evaluation_notes']}\n"
                else:
                    report += f"- Error: {test['response']}\n"
                report += "\n"
    
    # Save report
    report_file = output_path / "model_comparison_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save detailed JSON results
    json_file = output_path / "detailed_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n📊 Comparison report saved:")
    print(f"   📄 Summary: {report_file}")
    print(f"   📋 Detailed: {json_file}")


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description="Agentic Plot MVP - Single Model Testing")
    parser.add_argument("--models", nargs="+", 
                       choices=["qwen3_coder", "dark_reasoning", "qwq_planet", "venice", "all"],
                       default=["all"],
                       help="Models to test (default: all)")
    parser.add_argument("--input", help="Input markdown file for story generation")
    parser.add_argument("--creative-only", action="store_true", 
                       help="Run only creative writing tests, skip story generation")
    parser.add_argument("--output-dir", default="comparison_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = SingleModelAgent()
    available_models = agent.get_available_models()
    
    if not available_models:
        print("❌ No model files found in the specified directory")
        sys.exit(1)
    
    # Determine which models to test
    if "all" in args.models:
        models_to_test = list(available_models.keys())
    else:
        models_to_test = [m for m in args.models if m in available_models]
    
    if not models_to_test:
        print("❌ No valid models selected")
        sys.exit(1)
    
    print("🚀 Agentic Plot MVP - Single Model Testing")
    print("📊 Testing {len(models_to_test)} models: {', '.join(models_to_test)}")
    print("💾 Available VRAM: 32GB (RTX 5090)")
    
    if args.input and not args.creative_only:
        print(f"📖 Story input: {args.input}")
    else:
        print("🎨 Creative tests only")
    
    # Test each model
    all_results = []
    
    for i, model_name in enumerate(models_to_test, 1):
        print(f"\n🔄 Progress: {i}/{len(models_to_test)}")
        
        input_file = args.input if not args.creative_only else None
        result = run_single_model_test(model_name, agent, input_file)
        all_results.append(result)
        
        # Brief pause between models
        if i < len(models_to_test):
            print("⏳ Waiting 10 seconds before next model...")
            import time
            time.sleep(10)
    
    # Generate comparison report
    print("\n📊 Generating comparison report...")
    generate_comparison_report(all_results, args.output_dir)
    
    # Summary
    completed = sum(1 for r in all_results if r["status"] == "completed")
    print("\n✨ Testing Complete!")
    print(f"   Successfully tested: {completed}/{len(models_to_test)} models")
    print(f"   Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()