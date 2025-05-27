"""
Comprehensive Carbon Research Module

This module provides functions for running large-scale comprehensive carbon research
across multiple model-dataset combinations with proper resource management and progress tracking.
"""

import time
import gc
import os
import json
from datetime import datetime
from typing import Dict, Any, List

import torch
import numpy as np

from .runner import EcoLyzer


def run_comprehensive_analysis(comprehensive_research_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive environmental analysis by executing each model-dataset pair individually

    This approach provides better:
    - Memory management (models loaded/unloaded per run)
    - Error isolation (one failure doesn't stop everything)
    - Progress checkpointing
    - Resource cleanup between runs

    Args:
        comprehensive_research_config: Configuration dictionary containing:
            - models: List of model configurations
            - datasets: List of dataset configurations  
            - project: Project name for tracking
            - output_dir: Directory for saving results
            - Additional GenBench configuration options

    Returns:
        Dict[str, Any]: Comprehensive research results with aggregated analysis
    """

    # Extract configuration
    models = comprehensive_research_config["models"]
    datasets = comprehensive_research_config["datasets"]

    # Initialize overall results container
    overall_results = {
        "research_metadata": {
            "start_time": time.time(),
            "start_datetime": datetime.now().isoformat(),
            "total_combinations": len(models) * len(datasets),
            "config": comprehensive_research_config,
                            "framework_version": "ML-EcoLyzer v1.0 - Individual Run Mode"
        },
        "individual_runs": {},
        "aggregated_analysis": {},
        "progress_log": [],
        "error_log": []
    }

    print(f"🚀 Starting Comprehensive Environmental Analysis")
    print(f"{'='*80}")
    print(f"📊 Total Combinations: {len(models)} models × {len(datasets)} datasets = {len(models) * len(datasets)} runs")
    print(f"⏱️ Estimated Duration: ~{(len(models) * len(datasets) * 5):.0f} minutes")
    print(f"💾 Results will be saved incrementally to avoid data loss")
    print(f"{'='*80}")

    combination_count = 0
    successful_runs = 0
    failed_runs = 0

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"carbon_analysis_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Iterate through each model-dataset combination
    for model_idx, model_config in enumerate(models):
        model_name = model_config["name"]
        model_short_name = model_name.split("/")[-1]  # Extract short name for logging

        for dataset_idx, dataset_config in enumerate(datasets):
            combination_count += 1
            dataset_name = dataset_config["name"]
            dataset_short_name = dataset_name.split("/")[-1]

            # Create unique run identifier
            run_id = f"run_{combination_count:03d}_{model_short_name}_{dataset_short_name}"

            print(f"\n🔬 Research Run {combination_count}/{len(models) * len(datasets)}")
            print(f"{'='*60}")
            print(f"🤖 Model: {model_name}")
            print(f"📚 Dataset: {dataset_name}")
            print(f"🆔 Run ID: {run_id}")
            print(f"{'='*60}")

            # Log progress
            progress_entry = {
                "run_id": run_id,
                "combination_count": combination_count,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "start_time": time.time(),
                "status": "starting"
            }
            overall_results["progress_log"].append(progress_entry)

            try:
                # Create individual configuration for this run
                individual_config = create_individual_run_config(
                    comprehensive_research_config,
                    model_config,
                    dataset_config,
                    run_id
                )

                # Clear GPU memory before starting new run
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Log GPU memory status
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
                    print(f"🔧 GPU Memory: {gpu_memory_allocated:.1f}GB allocated, {gpu_memory_cached:.1f}GB cached")

                # Initialize and run individual analysis
                print(f"⚡ Initializing EcoLyzer for {run_id}...")
                analyzer = EcoLyzer(individual_config)

                # Run the analysis
                print(f"🏃 Running analysis...")
                individual_results = analyzer.run()

                # Extract the actual result (should be single model-dataset pair)
                result_keys = [k for k in individual_results.keys() if not k.startswith('ERROR') and k != 'final_report']

                if result_keys:
                    # Success case
                    main_result_key = result_keys[0]
                    main_result = individual_results[main_result_key]

                    # Add run metadata
                    main_result["run_metadata"] = {
                        "run_id": run_id,
                        "combination_count": combination_count,
                        "individual_config": individual_config,
                        "run_duration_seconds": time.time() - progress_entry["start_time"]
                    }

                    # Store in overall results
                    overall_results["individual_runs"][run_id] = main_result
                    successful_runs += 1

                    # Log success
                    co2_emissions = main_result["emissions_analysis"]["total_kg_co2"]
                    efficiency_score = main_result.get("environmental_assessment", {}).get("integrated_assessment", {}).get("overall_efficiency_score", "N/A")

                    print(f"✅ Run {run_id} completed successfully!")
                    print(f"   💨 CO2 Emissions: {co2_emissions:.6f} kg")
                    print(f"   ⚡ Efficiency Score: {efficiency_score}")

                    # Update progress log
                    progress_entry["status"] = "completed"
                    progress_entry["end_time"] = time.time()
                    progress_entry["co2_emissions"] = co2_emissions
                    progress_entry["efficiency_score"] = efficiency_score

                else:
                    # No successful results
                    raise Exception("No successful results generated")

            except Exception as e:
                # Error handling
                failed_runs += 1
                error_msg = str(e)

                print(f"❌ Run {run_id} failed: {error_msg}")

                # Log error
                error_entry = {
                    "run_id": run_id,
                    "combination_count": combination_count,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "error": error_msg,
                    "timestamp": time.time()
                }
                overall_results["error_log"].append(error_entry)

                # Update progress log
                progress_entry["status"] = "failed"
                progress_entry["end_time"] = time.time()
                progress_entry["error"] = error_msg

                # Continue with next combination
                continue

            finally:
                # Cleanup after each run
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Save intermediate results every 3 runs
                if combination_count % 3 == 0:
                    save_intermediate_results(overall_results, results_dir, combination_count)

            # Brief pause between runs to allow system cleanup
            time.sleep(2)

    # Final processing and analysis
    print(f"\n🎉 Analysis Complete!")
    print(f"{'='*80}")
    print(f"✅ Successful runs: {successful_runs}/{len(models) * len(datasets)}")
    print(f"❌ Failed runs: {failed_runs}/{len(models) * len(datasets)}")
    print(f"📊 Success rate: {(successful_runs/(len(models) * len(datasets)))*100:.1f}%")

    # Generate aggregated analysis
    overall_results["aggregated_analysis"] = generate_aggregated_analysis(overall_results["individual_runs"])

    # Add completion metadata
    overall_results["research_metadata"]["end_time"] = time.time()
    overall_results["research_metadata"]["end_datetime"] = datetime.now().isoformat()
    overall_results["research_metadata"]["total_duration_seconds"] = overall_results["research_metadata"]["end_time"] - overall_results["research_metadata"]["start_time"]
    overall_results["research_metadata"]["successful_runs"] = successful_runs
    overall_results["research_metadata"]["failed_runs"] = failed_runs
    overall_results["research_metadata"]["success_rate"] = successful_runs / (len(models) * len(datasets))

    # Save final results
    save_final_results(overall_results, results_dir)

    print(f"💾 Final results saved to: {results_dir}/")
    print(f"{'='*80}")

    return overall_results


def create_individual_run_config(base_config: Dict[str, Any], model_config: Dict[str, Any],
                                dataset_config: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """Create configuration for individual model-dataset run"""

    os.makedirs(f"individual_runs/{run_id}", exist_ok=True)

    individual_config = {
        # Copy base configuration
        **base_config,

        # Override project name for this specific run
        "project": f"{base_config['project']}_{run_id}",

        # Single model and dataset for this run
        "models": [model_config],
        "datasets": [dataset_config],

        # Add run-specific metadata
        "run_metadata": {
            "run_id": run_id,
            "individual_run_mode": True,
            "base_project": base_config["project"]
        },

        # Adjust output paths to avoid conflicts
        "output_dir": f"individual_runs/{run_id}",
        "emissions_file": f"{run_id}_emissions.csv"
    }

    return individual_config


def save_intermediate_results(overall_results: Dict[str, Any], results_dir: str, run_count: int):
    """Save intermediate results to prevent data loss"""
    try:
        intermediate_file = os.path.join(results_dir, f"intermediate_results_run_{run_count:03d}.json")

        # Create a lightweight version for intermediate saves
        intermediate_data = {
            "research_metadata": overall_results["research_metadata"],
            "progress_summary": {
                "total_runs_attempted": run_count,
                "successful_runs": len(overall_results["individual_runs"]),
                "failed_runs": len(overall_results["error_log"]),
                "latest_results_keys": list(overall_results["individual_runs"].keys())[-5:]  # Last 5 runs
            },
            "error_summary": overall_results["error_log"]
        }

        with open(intermediate_file, "w") as f:
            json.dump(intermediate_data, f, indent=2, default=str)

        print(f"💾 Intermediate results saved: {intermediate_file}")

    except Exception as e:
        print(f"⚠️ Failed to save intermediate results: {e}")


def generate_aggregated_analysis(individual_runs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate aggregated analysis across all successful runs"""

    if not individual_runs:
        return {"error": "No successful runs to analyze"}

    # Extract metrics from all runs
    co2_emissions = []
    efficiency_scores = []
    model_performance = {}
    dataset_performance = {}

    for run_id, result in individual_runs.items():
        # CO2 emissions
        co2 = result["emissions_analysis"]["total_kg_co2"]
        co2_emissions.append(co2)

        # Efficiency scores
        efficiency = result.get("environmental_assessment", {}).get("integrated_assessment", {}).get("overall_efficiency_score", 0)
        efficiency_scores.append(efficiency)

        # Model performance tracking
        model_name = result["model_name"]
        if model_name not in model_performance:
            model_performance[model_name] = {"co2_emissions": [], "efficiency_scores": []}
        model_performance[model_name]["co2_emissions"].append(co2)
        model_performance[model_name]["efficiency_scores"].append(efficiency)

        # Dataset performance tracking
        dataset_name = result["dataset_name"]
        if dataset_name not in dataset_performance:
            dataset_performance[dataset_name] = {"co2_emissions": [], "efficiency_scores": []}
        dataset_performance[dataset_name]["co2_emissions"].append(co2)
        dataset_performance[dataset_name]["efficiency_scores"].append(efficiency)

    # Calculate aggregated statistics
    aggregated_analysis = {
        "overall_statistics": {
            "total_successful_runs": len(individual_runs),
            "total_co2_emissions_kg": sum(co2_emissions),
            "average_co2_per_run_kg": sum(co2_emissions) / len(co2_emissions),
            "co2_emissions_range": (min(co2_emissions), max(co2_emissions)),
            "average_efficiency_score": sum(efficiency_scores) / len(efficiency_scores),
            "efficiency_score_range": (min(efficiency_scores), max(efficiency_scores))
        },

        "model_analysis": {
            model: {
                "total_co2_kg": sum(data["co2_emissions"]),
                "avg_co2_per_run_kg": sum(data["co2_emissions"]) / len(data["co2_emissions"]),
                "avg_efficiency_score": sum(data["efficiency_scores"]) / len(data["efficiency_scores"]),
                "run_count": len(data["co2_emissions"])
            }
            for model, data in model_performance.items()
        },

        "dataset_analysis": {
            dataset: {
                "total_co2_kg": sum(data["co2_emissions"]),
                "avg_co2_per_run_kg": sum(data["co2_emissions"]) / len(data["co2_emissions"]),
                "avg_efficiency_score": sum(data["efficiency_scores"]) / len(data["efficiency_scores"]),
                "run_count": len(data["co2_emissions"])
            }
            for dataset, data in dataset_performance.items()
        },

        "research_insights": generate_research_insights(model_performance, dataset_performance, co2_emissions, efficiency_scores)
    }

    return aggregated_analysis


def generate_research_insights(model_performance: Dict, dataset_performance: Dict,
                             co2_emissions: List[float], efficiency_scores: List[float]) -> List[str]:
    """Generate research insights from aggregated data"""
    insights = []

    # Model efficiency insights
    if len(model_performance) > 1:
        best_model = min(model_performance.keys(),
                        key=lambda m: sum(model_performance[m]["co2_emissions"]) / len(model_performance[m]["co2_emissions"]))
        worst_model = max(model_performance.keys(),
                         key=lambda m: sum(model_performance[m]["co2_emissions"]) / len(model_performance[m]["co2_emissions"]))

        insights.append(f"Most carbon-efficient model: {best_model}")
        insights.append(f"Least carbon-efficient model: {worst_model}")

    # Dataset complexity insights
    if len(dataset_performance) > 1:
        most_demanding_dataset = max(dataset_performance.keys(),
                                   key=lambda d: sum(dataset_performance[d]["co2_emissions"]) / len(dataset_performance[d]["co2_emissions"]))
        insights.append(f"Most computationally demanding dataset: {most_demanding_dataset}")

    # Overall efficiency insights
    high_efficiency_runs = sum(1 for score in efficiency_scores if score > 0.7)
    insights.append(f"High efficiency runs (>0.7): {high_efficiency_runs}/{len(efficiency_scores)} ({(high_efficiency_runs/len(efficiency_scores)*100):.1f}%)")

    # Carbon footprint insights
    total_co2 = sum(co2_emissions)
    insights.append(f"Total carbon footprint: {total_co2:.6f} kg CO2")
    insights.append(f"Equivalent to driving: {total_co2 * 2.31:.2f} km in average car")  # EPA conversion factor

    return insights


def save_final_results(overall_results: Dict[str, Any], results_dir: str):
    """Save final comprehensive results"""

    # Save full results
    full_results_file = os.path.join(results_dir, "comprehensive_carbon_analysis_results.json")
    with open(full_results_file, "w") as f:
        json.dump(overall_results, f, indent=2, default=str)

    # Save summary CSV if pandas available
    try:
        import pandas as pd

        # Create summary dataframe
        summary_data = []
        for run_id, result in overall_results["individual_runs"].items():
            summary_row = {
                "run_id": run_id,
                "model_name": result["model_name"],
                "dataset_name": result["dataset_name"],
                "co2_emissions_kg": result["emissions_analysis"]["total_kg_co2"],
                "efficiency_score": result.get("environmental_assessment", {}).get("integrated_assessment", {}).get("overall_efficiency_score", 0),
                "samples_processed": result.get("successful_predictions", 0),
                "run_duration_seconds": result.get("run_metadata", {}).get("run_duration_seconds", 0)
            }
            summary_data.append(summary_row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = os.path.join(results_dir, "analysis_summary.csv")
            df.to_csv(csv_file, index=False)
            print(f"📊 Summary CSV saved: {csv_file}")

    except ImportError:
        print("⚠️ pandas not available - skipping CSV export")


def analyze_research_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze comprehensive research results and provide insights
    
    Args:
        results: Results dictionary from run_comprehensive_analysis
        
    Returns:
        Dict with detailed analysis and recommendations
    """
    if "aggregated_analysis" not in results:
        return {"error": "No aggregated analysis found in results"}
    
    analysis = results["aggregated_analysis"]
    
    # Extract key metrics
    total_emissions = analysis["overall_statistics"]["total_co2_emissions_kg"]
    avg_emissions = analysis["overall_statistics"]["average_co2_per_run_kg"]
    avg_efficiency = analysis["overall_statistics"]["average_efficiency_score"]
    
    # Generate recommendations
    recommendations = []
    
    if avg_emissions > 0.01:  # High emissions threshold
        recommendations.append("Consider model optimization or quantization to reduce carbon footprint")
    
    if avg_efficiency < 0.5:  # Low efficiency threshold
        recommendations.append("Investigate hardware optimization or alternative deployment strategies")
    
    # Model comparison
    model_analysis = analysis.get("model_analysis", {})
    if len(model_analysis) > 1:
        best_model = min(model_analysis.keys(), 
                        key=lambda m: model_analysis[m]["avg_co2_per_run_kg"])
        recommendations.append(f"Consider using {best_model} for lower carbon impact")
    
    return {
        "summary": {
            "total_emissions_kg": total_emissions,
            "average_emissions_per_run_kg": avg_emissions,
            "average_efficiency_score": avg_efficiency,
            "total_runs": analysis["overall_statistics"]["total_successful_runs"]
        },
        "recommendations": recommendations,
        "insights": analysis.get("research_insights", []),
        "model_ranking": sorted(
            model_analysis.items(),
            key=lambda x: x[1]["avg_co2_per_run_kg"]
        ) if model_analysis else []
    }