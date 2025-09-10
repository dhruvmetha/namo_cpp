#!/usr/bin/env python3
"""Demo script showing failure classification in action with collection scripts."""

import os
import sys
import json
from pathlib import Path

# Add idfs directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'idfs'))

from idfs.failure_codes import FailureCode, FailureClassifier, get_failure_statistics

def analyze_collection_results(results_dir: str):
    """Analyze failure patterns from collection results directory."""
    print(f"🔍 Analyzing collection results in: {results_dir}")
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Directory not found: {results_dir}")
        return
    
    # Look for pickle files and summary files
    pkl_files = list(results_path.glob("*_results.pkl"))
    summary_files = list(results_path.glob("collection_summary_*.pkl"))
    
    print(f"📁 Found {len(pkl_files)} episode files and {len(summary_files)} summary files")
    
    if summary_files:
        # If we have summary files, show the failure analysis from there
        import pickle
        
        for summary_file in summary_files:
            with open(summary_file, 'rb') as f:
                summary = pickle.load(f)
            
            print(f"\n📊 Analysis from {summary_file.name}:")
            print(f"  Algorithm: {summary['collection_metadata']['algorithm']}")
            print(f"  Execution mode: {summary['collection_metadata'].get('execution_mode', 'unknown')}")
            
            if 'failure_analysis' in summary:
                failure_stats = summary['failure_analysis']
                print(f"  Total episodes: {failure_stats['total_episodes']}")
                print(f"  Success rate: {failure_stats['success_rate']:.1f}%")
                print(f"  Failed episodes: {failure_stats['failed_episodes']}")
                
                if failure_stats['failure_breakdown']:
                    print("  \n🚨 Top failure reasons:")
                    sorted_failures = sorted(
                        failure_stats['failure_breakdown'].items(), 
                        key=lambda x: x[1]['count'], 
                        reverse=True
                    )
                    for desc, info in sorted_failures:
                        print(f"    • {desc}: {info['count']} episodes ({info['percentage']:.1f}%)")
    
    else:
        print("ℹ️  No summary files found. Use the enhanced collection scripts to get failure analysis.")

def show_failure_code_reference():
    """Show reference of all available failure codes."""
    print("\n📚 Failure Code Reference:")
    print("=" * 60)
    
    for code in FailureCode:
        if code != FailureCode.SUCCESS:  # Skip success code
            desc = FailureClassifier.get_failure_description(code)
            print(f"  {code.value:2d} - {code.name:25} : {desc}")

def demonstrate_classification():
    """Demonstrate failure classification with example messages."""
    print("\n🎭 Failure Classification Demo:")
    print("=" * 60)
    
    example_errors = [
        "Planning timeout exceeded after 300.5 seconds",
        "No reachable edge idx found for object box_3",
        "No reachable edges at iteration 5", 
        "Robot-object collision detected during push manipulation",
        "Maximum search depth of 5 reached without finding solution",
        "Terminal state check limit of 5000 exceeded",
        "Failed to load PyTorch model from /path/to/model.pth",
        "MPC execution failed during object push",
        "Environment reset failed due to MuJoCo error",
        "Out of memory during state observation extraction",
        "Unknown error in planning system"
    ]
    
    for error_msg in example_errors:
        failure_code = FailureClassifier.classify_failure(error_msg)
        description = FailureClassifier.get_failure_description(failure_code)
        print(f"  {failure_code.value:2d} | {error_msg}")
        print(f"     └─ {description}")
        print()

def show_usage_examples():
    """Show how to use the enhanced collection scripts."""
    print("\n💡 Usage Examples:")
    print("=" * 60)
    
    print("🔹 Sequential ML Collection with Failure Analysis:")
    print("   python python/idfs/sequential_ml_collection.py \\")
    print("     --algorithm idfs \\")
    print("     --goal-strategy ml \\")
    print("     --epsilon 0.3 \\")
    print("     --ml-goal-model /path/to/model \\")
    print("     --output-dir /tmp/results \\")
    print("     --start-idx 0 --end-idx 5 \\")
    print("     --verbose")
    print()
    
    print("🔹 Parallel Collection with Failure Analysis:")
    print("   python python/idfs/modular_parallel_collection.py \\")
    print("     --algorithm idfs \\")
    print("     --goal-strategy ml \\")
    print("     --epsilon 0.2 \\")
    print("     --ml-goal-model /path/to/model \\")
    print("     --output-dir /tmp/results \\") 
    print("     --start-idx 0 --end-idx 10 \\")
    print("     --workers 4 \\")
    print("     --verbose")
    print()
    
    print("📈 Output includes:")
    print("  • Detailed failure classification for each episode")
    print("  • Summary with failure statistics and breakdowns")
    print("  • Human-readable failure descriptions")
    print("  • Top failure reasons with percentages")

def main():
    """Main demo function."""
    print("🎯 NAMO Failure Analysis Demo")
    print("=" * 80)
    
    # Show failure code reference
    show_failure_code_reference()
    
    # Demonstrate classification
    demonstrate_classification()
    
    # Show usage examples
    show_usage_examples()
    
    # Try to analyze existing results if any
    possible_dirs = [
        "/tmp/sequential_ml_results",
        "/tmp/modular_ml_results", 
        "./sequential_data",
        "./modular_data"
    ]
    
    print("\n🔍 Looking for existing collection results...")
    found_results = False
    for results_dir in possible_dirs:
        if os.path.exists(results_dir):
            analyze_collection_results(results_dir)
            found_results = True
    
    if not found_results:
        print("ℹ️  No existing collection results found.")
        print("   Run the enhanced collection scripts to generate failure analysis!")
    
    print(f"\n{'='*80}")
    print("✅ Demo complete! The failure classification system is now integrated")
    print("   into both sequential_ml_collection.py and modular_parallel_collection.py")
    print("\n🎁 Key Benefits:")
    print("  • Systematic classification of 17 different failure types")
    print("  • Automatic pattern matching from error messages")  
    print("  • Statistical analysis of failure patterns")
    print("  • Better debugging and performance analysis")
    print("  • Consistent failure tracking across collection runs")

if __name__ == "__main__":
    main()