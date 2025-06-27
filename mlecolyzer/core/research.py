"""
Comprehensive Carbon and Water Research Module

This module provides functions for running large-scale comprehensive environmental research
across multiple model-dataset combinations with proper resource management, progress tracking,
and comprehensive water footprint analysis.
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
from ..utils.parameters import ParameterEstimator

def run_comprehensive_analysis(comprehensive_research_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive environmental analysis by executing each model-dataset pair individually

    This approach provides better:
    - Memory management (models loaded/unloaded per run)
    - Error isolation (one failure doesn't stop everything)
    - Progress checkpointing
    - Resource cleanup between runs
    - Comprehensive water footprint tracking

    Args:
        comprehensive_research_config: Configuration dictionary containing:
            - models: List of model configurations
            - datasets: List of dataset configurations  
            - project: Project name for tracking
            - output_dir: Directory for saving results
            - Additional EcoLyzer configuration options

    Returns:
        Dict[str, Any]: Comprehensive research results with aggregated analysis including water footprint
    """

    # Extract configuration
    models = comprehensive_research_config["models"]
    datasets = comprehensive_research_config["datasets"]

    # Initialize parameter estimator for ESS analysis
    parameter_estimator = ParameterEstimator()    

    # Initialize overall results container
    overall_results = {
        "research_metadata": {
            "start_time": time.time(),
            "start_datetime": datetime.now().isoformat(),
            "total_combinations": len(models) * len(datasets),
            "config": comprehensive_research_config,
            "framework_version": "ML-EcoLyzer v1.0 - Individual Run Mode with Water Footprint Tracking"
        },
        "individual_runs": {},
        "aggregated_analysis": {},
        "progress_log": [],
        "error_log": []
    }

    print(f"🚀 Starting Comprehensive Environmental Analysis with Water Footprint Tracking")
    print(f"{'='*80}")
    print(f"📊 Total Combinations: {len(models)} models × {len(datasets)} datasets = {len(models) * len(datasets)} runs")
    print(f"⏱️ Estimated Duration: ~{(len(models) * len(datasets) * 5):.0f} minutes")
    print(f"💾 Results will be saved incrementally to avoid data loss")
    print(f"💧 Water footprint tracking enabled for all runs")
    print(f"{'='*80}")

    combination_count = 0
    successful_runs = 0
    failed_runs = 0

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"environmental_analysis_results_{timestamp}"
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
            print(f"💧 Water tracking: Enabled")
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

                    # Log success with water footprint
                    co2_emissions = main_result["emissions_analysis"]["total_kg_co2"]
                    water_liters = main_result["water_analysis"]["total_water_liters"]
                    water_bottles = main_result["water_analysis"]["water_equivalent_bottles"]
                    efficiency_score = main_result.get("environmental_assessment", {}).get("integrated_assessment", {}).get("overall_efficiency_score", "N/A")
                    water_efficiency = main_result.get("environmental_assessment", {}).get("water_analysis", {}).get("water_efficiency", "N/A")

                    print(f"✅ Run {run_id} completed successfully!")
                    print(f"   💨 CO2 Emissions: {co2_emissions:.6f} kg")
                    print(f"   💧 Water Footprint: {water_liters:.3f} L ({water_bottles:.1f} bottles)")
                    print(f"   ⚡ Overall Efficiency: {efficiency_score}")
                    print(f"   🌊 Water Efficiency: {water_efficiency}")

                    # Update progress log
                    progress_entry["status"] = "completed"
                    progress_entry["end_time"] = time.time()
                    progress_entry["co2_emissions"] = co2_emissions
                    progress_entry["water_liters"] = water_liters
                    progress_entry["efficiency_score"] = efficiency_score
                    progress_entry["water_efficiency"] = water_efficiency

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

    # Generate aggregated analysis including water footprint
    overall_results["aggregated_analysis"] = generate_aggregated_analysis(
        overall_results["individual_runs"], 
        parameter_estimator
    )

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
    print(f"💧 Water footprint analysis included in all results")
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
            "base_project": base_config["project"],
            "water_tracking_enabled": True
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
            "error_summary": overall_results["error_log"],
            "water_tracking_enabled": True
        }

        with open(intermediate_file, "w") as f:
            json.dump(intermediate_data, f, indent=2, default=str)

        print(f"💾 Intermediate results saved: {intermediate_file}")

    except Exception as e:
        print(f"⚠️ Failed to save intermediate results: {e}")


def generate_aggregated_analysis(individual_runs: Dict[str, Any], 
                                        parameter_estimator: ParameterEstimator) -> Dict[str, Any]:
    """Generate aggregated analysis across all successful runs including water footprint"""

    if not individual_runs:
        return {"error": "No successful runs to analyze"}

    # Extract metrics from all runs
    co2_emissions = []
    water_consumption = []
    efficiency_scores = []
    water_efficiency_scores = []
    model_performance = {}
    dataset_performance = {}

    model_parameters = {}
    model_ess_scores = {}
    parameter_sources = {}
    framework_ess = {}


    for run_id, result in individual_runs.items():

        model_name = result.get("model_name", "unknown")
        framework = result.get("framework", "unknown")
        
        # Parameter analysis
        model_analysis = result.get("model_analysis", {})
        param_count = model_analysis.get("parameter_count", 0)
        param_source = model_analysis.get("parameter_source", "unknown")
        
        model_parameters[model_name] = param_count
        
        if param_source not in parameter_sources:
            parameter_sources[param_source] = 0
        parameter_sources[param_source] += 1
        
        # ESS analysis
        sustainability = result.get("sustainability_metrics", {})
        if "ess_score" in sustainability:
            ess_score = sustainability["ess_score"]
            model_ess_scores[model_name] = ess_score
            
            if framework not in framework_ess:
                framework_ess[framework] = []
            framework_ess[framework].append(ess_score)

        # CO2 emissions
        co2 = result["emissions_analysis"]["total_kg_co2"]
        co2_emissions.append(co2)

        # Water consumption
        water = result["water_analysis"]["total_water_liters"]
        water_consumption.append(water)

        # Efficiency scores
        efficiency = result.get("environmental_assessment", {}).get("integrated_assessment", {}).get("overall_efficiency_score", 0)
        efficiency_scores.append(efficiency)

        # Water efficiency scores
        water_eff = result.get("environmental_assessment", {}).get("water_analysis", {}).get("water_efficiency", 0.5)
        water_efficiency_scores.append(water_eff)

        # Model performance tracking
        model_name = result["model_name"]
        if model_name not in model_performance:
            model_performance[model_name] = {
                "co2_emissions": [], "water_consumption": [], 
                "efficiency_scores": [], "water_efficiency_scores": []
            }
        model_performance[model_name]["co2_emissions"].append(co2)
        model_performance[model_name]["water_consumption"].append(water)
        model_performance[model_name]["efficiency_scores"].append(efficiency)
        model_performance[model_name]["water_efficiency_scores"].append(water_eff)

        # Dataset performance tracking
        dataset_name = result["dataset_name"]
        if dataset_name not in dataset_performance:
            dataset_performance[dataset_name] = {
                "co2_emissions": [], "water_consumption": [], 
                "efficiency_scores": [], "water_efficiency_scores": []
            }
        dataset_performance[dataset_name]["co2_emissions"].append(co2)
        dataset_performance[dataset_name]["water_consumption"].append(water)
        dataset_performance[dataset_name]["efficiency_scores"].append(efficiency)
        dataset_performance[dataset_name]["water_efficiency_scores"].append(water_eff)

    # Calculate aggregated statistics
    aggregated_analysis = {
        "overall_statistics": {
            "total_successful_runs": len(individual_runs),
            "total_co2_emissions_kg": sum(co2_emissions),
            "average_co2_per_run_kg": sum(co2_emissions) / len(co2_emissions),
            "co2_emissions_range": (min(co2_emissions), max(co2_emissions)),
            "total_water_liters": sum(water_consumption),
            "average_water_per_run_liters": sum(water_consumption) / len(water_consumption),
            "water_consumption_range": (min(water_consumption), max(water_consumption)),
            "water_bottles_equivalent": sum(water_consumption) / 0.5,
            "average_efficiency_score": sum(efficiency_scores) / len(efficiency_scores),
            "efficiency_score_range": (min(efficiency_scores), max(efficiency_scores)),
            "average_water_efficiency_score": sum(water_efficiency_scores) / len(water_efficiency_scores),
            "water_efficiency_range": (min(water_efficiency_scores), max(water_efficiency_scores))
        },

        "model_analysis": {
            model: {
                "total_co2_kg": sum(data["co2_emissions"]),
                "avg_co2_per_run_kg": sum(data["co2_emissions"]) / len(data["co2_emissions"]),
                "total_water_liters": sum(data["water_consumption"]),
                "avg_water_per_run_liters": sum(data["water_consumption"]) / len(data["water_consumption"]),
                "water_bottles_equivalent": sum(data["water_consumption"]) / 0.5,
                "avg_efficiency_score": sum(data["efficiency_scores"]) / len(data["efficiency_scores"]),
                "avg_water_efficiency_score": sum(data["water_efficiency_scores"]) / len(data["water_efficiency_scores"]),
                "run_count": len(data["co2_emissions"])
            }
            for model, data in model_performance.items()
        },

        "dataset_analysis": {
            dataset: {
                "total_co2_kg": sum(data["co2_emissions"]),
                "avg_co2_per_run_kg": sum(data["co2_emissions"]) / len(data["co2_emissions"]),
                "total_water_liters": sum(data["water_consumption"]),
                "avg_water_per_run_liters": sum(data["water_consumption"]) / len(data["water_consumption"]),
                "water_bottles_equivalent": sum(data["water_consumption"]) / 0.5,
                "avg_efficiency_score": sum(data["efficiency_scores"]) / len(data["efficiency_scores"]),
                "avg_water_efficiency_score": sum(data["water_efficiency_scores"]) / len(data["water_efficiency_scores"]),
                "run_count": len(data["co2_emissions"])
            }
            for dataset, data in dataset_performance.items()
        },

        "water_footprint_analysis": {
            "total_water_consumption_liters": sum(water_consumption),
            "average_water_per_model_liters": sum(water_consumption) / len(water_consumption),
            "water_bottles_equivalent": sum(water_consumption) / 0.5,
            "water_gallons_equivalent": sum(water_consumption) / 3.785,
            "most_water_efficient_model": min(model_performance.keys(), 
                key=lambda m: sum(model_performance[m]["water_consumption"]) / len(model_performance[m]["water_consumption"])),
            "least_water_efficient_model": max(model_performance.keys(), 
                key=lambda m: sum(model_performance[m]["water_consumption"]) / len(model_performance[m]["water_consumption"])),
            "water_efficiency_improvement_potential": calculate_water_efficiency_potential(model_performance)
        },

        # Comprehensive parameter analysis
        "parameter_analysis": {
            "total_parameters_analyzed": sum(model_parameters.values()),
            "average_parameters_per_model": sum(model_parameters.values()) / len(model_parameters) if model_parameters else 0,
            "parameter_distribution": {
                "model_parameter_counts": model_parameters,
                "parameter_sources": parameter_sources,
                "size_categories": _categorize_models_by_size(model_parameters)
            },
            "largest_model": max(model_parameters.items(), key=lambda x: x[1]) if model_parameters else None,
            "smallest_model": min(model_parameters.items(), key=lambda x: x[1]) if model_parameters else None,
            "parameter_range": (min(model_parameters.values()), max(model_parameters.values())) if model_parameters else (0, 0)
        },
        
        # Comprehensive ESS analysis
        "ess_analysis": _generate_comprehensive_ess_analysis(
            model_ess_scores, framework_ess, parameter_estimator, individual_runs
        ),


        "research_insights": generate_research_insights(model_performance, dataset_performance, co2_emissions, water_consumption, efficiency_scores, water_efficiency_scores, model_parameters, model_ess_scores)
    }

    ## ESS comparative analysis
    ess_scores = [(run_id, result["sustainability_metrics"]["ess_score"]) 
                  for run_id, result in individual_runs.items()]
    
    aggregated_analysis["ess_analysis"] = {
        "model_ranking_by_ess": sorted(ess_scores, key=lambda x: x[1], reverse=True),
        "ess_distribution": _calculate_ess_distribution(ess_scores)
    }    
    

    return aggregated_analysis

def calculate_water_efficiency_potential(model_performance: Dict[str, Any]) -> Dict[str, float]:
    """Calculate potential water efficiency improvements"""
    if len(model_performance) < 2:
        return {"improvement_potential_percent": 0.0}
    
    avg_water_by_model = {
        model: sum(data["water_consumption"]) / len(data["water_consumption"])
        for model, data in model_performance.items()
    }
    
    best_water = min(avg_water_by_model.values())
    worst_water = max(avg_water_by_model.values())
    
    if worst_water > 0:
        improvement_potential = ((worst_water - best_water) / worst_water) * 100
        return {
            "improvement_potential_percent": improvement_potential,
            "water_saved_liters": worst_water - best_water,
            "bottles_saved": (worst_water - best_water) / 0.5
        }
    
    return {"improvement_potential_percent": 0.0}

def generate_research_insights(model_performance: Dict, dataset_performance: Dict,
                                       co2_emissions: List[float], water_consumption: List[float],
                                       efficiency_scores: List[float], water_efficiency_scores: List[float],
                                       model_parameters: Dict[str, int], model_ess_scores: Dict[str, float]) -> List[str]:
    """Generate research insights from aggregated data including water footprint"""
    insights = []

    # Model efficiency insights (CO2)
    if len(model_performance) > 1:
        best_co2_model = min(model_performance.keys(),
                            key=lambda m: sum(model_performance[m]["co2_emissions"]) / len(model_performance[m]["co2_emissions"]))
        worst_co2_model = max(model_performance.keys(),
                             key=lambda m: sum(model_performance[m]["co2_emissions"]) / len(model_performance[m]["co2_emissions"]))

        insights.append(f"Most carbon-efficient model: {best_co2_model}")
        insights.append(f"Least carbon-efficient model: {worst_co2_model}")

    # Model water efficiency insights
    if len(model_performance) > 1:
        best_water_model = min(model_performance.keys(),
                              key=lambda m: sum(model_performance[m]["water_consumption"]) / len(model_performance[m]["water_consumption"]))
        worst_water_model = max(model_performance.keys(),
                               key=lambda m: sum(model_performance[m]["water_consumption"]) / len(model_performance[m]["water_consumption"]))

        insights.append(f"Most water-efficient model: {best_water_model}")
        insights.append(f"Least water-efficient model: {worst_water_model}")

        # Water efficiency comparison
        avg_best_water = sum(model_performance[best_water_model]["water_consumption"]) / len(model_performance[best_water_model]["water_consumption"])
        avg_worst_water = sum(model_performance[worst_water_model]["water_consumption"]) / len(model_performance[worst_water_model]["water_consumption"])
        
        if avg_worst_water > 0:
            water_improvement = ((avg_worst_water - avg_best_water) / avg_worst_water) * 100
            insights.append(f"Water efficiency improvement potential: {water_improvement:.1f}% by choosing best model")            

    # Dataset complexity insights
    if len(dataset_performance) > 1:
        most_demanding_dataset_co2 = max(dataset_performance.keys(),
                                        key=lambda d: sum(dataset_performance[d]["co2_emissions"]) / len(dataset_performance[d]["co2_emissions"]))
        most_demanding_dataset_water = max(dataset_performance.keys(),
                                          key=lambda d: sum(dataset_performance[d]["water_consumption"]) / len(dataset_performance[d]["water_consumption"]))
        
        insights.append(f"Most computationally demanding dataset (CO2): {most_demanding_dataset_co2}")
        insights.append(f"Most water-intensive dataset: {most_demanding_dataset_water}")

    # Overall efficiency insights
    high_efficiency_runs = sum(1 for score in efficiency_scores if score > 0.7)
    high_water_efficiency_runs = sum(1 for score in water_efficiency_scores if score > 0.7)
    
    insights.append(f"High overall efficiency runs (>0.7): {high_efficiency_runs}/{len(efficiency_scores)} ({(high_efficiency_runs/len(efficiency_scores)*100):.1f}%)")
    insights.append(f"High water efficiency runs (>0.7): {high_water_efficiency_runs}/{len(water_efficiency_scores)} ({(high_water_efficiency_runs/len(water_efficiency_scores)*100):.1f}%)")

    # Carbon footprint insights
    total_co2 = sum(co2_emissions)
    insights.append(f"Total carbon footprint: {total_co2:.6f} kg CO2")
    insights.append(f"Equivalent to driving: {total_co2 * 2.31:.2f} km in average car")  # EPA conversion factor

    # Water footprint insights
    total_water = sum(water_consumption)
    avg_water = total_water / len(water_consumption)
    insights.append(f"Total water footprint: {total_water:.3f} L ({total_water/0.5:.1f} bottles)")
    insights.append(f"Average water per run: {avg_water:.3f} L")
    
    # Water consumption context
    if total_water > 10:
        insights.append(f"Water consumption equivalent to {total_water/9.5:.1f} minutes of showering")
    if total_water > 2:
        insights.append(f"Water consumption represents {total_water/2.0:.1f} days of drinking water")

    # Efficiency correlation
    if len(efficiency_scores) == len(water_efficiency_scores):
        correlation = np.corrcoef(efficiency_scores, water_efficiency_scores)[0, 1]
        if not np.isnan(correlation):
            insights.append(f"Overall efficiency and water efficiency correlation: {correlation:.3f}")

    # Parameter efficiency insights
    if model_parameters:
        total_params = sum(model_parameters.values())
        avg_params = total_params / len(model_parameters)
        
        insights.append(f"Total parameters analyzed: {total_params:,} across {len(model_parameters)} models")
        insights.append(f"Average model size: {avg_params:,.0f} parameters")
        
        # Parameter distribution
        large_models = sum(1 for p in model_parameters.values() if p > 1_000_000_000)
        if large_models > 0:
            insights.append(f"{large_models}/{len(model_parameters)} models have >1B parameters")
    
    # ESS insights
    if model_ess_scores:
        avg_ess = sum(model_ess_scores.values()) / len(model_ess_scores)
        efficient_models = sum(1 for ess in model_ess_scores.values() if ess >= 0.5)
        
        insights.append(f"Average ESS score: {avg_ess:.3f}")
        insights.append(f"Efficient models (ESS ≥ 0.5): {efficient_models}/{len(model_ess_scores)}")
        
        if avg_ess < 0.2:
            insights.append("Overall low ESS scores suggest need for model optimization strategies")
        elif avg_ess > 0.5:
            insights.append("Good overall ESS performance indicates efficient model selection")
        
        # Best ESS model
        best_ess_model = max(model_ess_scores.items(), key=lambda x: x[1])
        insights.append(f"Most environmentally efficient: {best_ess_model[0]} (ESS: {best_ess_model[1]:.3f})")
    
    # Parameter-CO2 efficiency
    if model_parameters and co2_emissions:
        total_co2 = sum(co2_emissions)
        param_co2_ratio = total_params / (total_co2 * 1000) if total_co2 > 0 else 0
        insights.append(f"Overall parameter efficiency: {param_co2_ratio:,.0f} parameters per gram CO₂")
    
    return insights


def save_final_results(overall_results: Dict[str, Any], results_dir: str):
    """Save final comprehensive results including water footprint analysis"""

    # Save full results
    full_results_file = os.path.join(results_dir, "comprehensive_environmental_analysis_results.json")
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
                "framework": result.get("framework", "unknown"),
                "co2_emissions_kg": result["emissions_analysis"]["total_kg_co2"],
                "water_consumption_liters": result["water_analysis"]["total_water_liters"],
                "water_bottles_equivalent": result["water_analysis"]["water_equivalent_bottles"],
                "efficiency_score": result.get("environmental_assessment", {}).get("integrated_assessment", {}).get("overall_efficiency_score", 0),
                "water_efficiency_score": result.get("environmental_assessment", {}).get("water_analysis", {}).get("water_efficiency", 0),
                "samples_processed": result.get("successful_predictions", 0),
                "run_duration_seconds": result.get("run_metadata", {}).get("run_duration_seconds", 0),
                "region": result["water_analysis"].get("region", "unknown"),
                "water_intensity_factor": result["water_analysis"].get("water_intensity_factor", 0)
            }
            summary_data.append(summary_row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = os.path.join(results_dir, "environmental_analysis_summary.csv")
            df.to_csv(csv_file, index=False)
            print(f"📊 Summary CSV saved: {csv_file}")

            # Create water-focused summary
            water_summary = df[['model_name', 'water_consumption_liters', 'water_bottles_equivalent', 
                               'water_efficiency_score', 'region', 'water_intensity_factor']].copy()
            water_csv_file = os.path.join(results_dir, "water_footprint_summary.csv")
            water_summary.to_csv(water_csv_file, index=False)
            print(f"💧 Water footprint CSV saved: {water_csv_file}")

    except ImportError:
        print("⚠️ pandas not available - skipping CSV export")


def _categorize_models_by_size(model_parameters: Dict[str, int]) -> Dict[str, Any]:
    """Categorize models by parameter count"""
    size_categories = {
        "tiny": {"threshold": 10_000_000, "models": []},      # <10M
        "small": {"threshold": 100_000_000, "models": []},    # 10M-100M
        "medium": {"threshold": 1_000_000_000, "models": []}, # 100M-1B
        "large": {"threshold": 10_000_000_000, "models": []}, # 1B-10B
        "xlarge": {"threshold": float('inf'), "models": []}   # >10B
    }
    
    for model, param_count in model_parameters.items():
        if param_count < size_categories["tiny"]["threshold"]:
            size_categories["tiny"]["models"].append({"model": model, "parameters": param_count})
        elif param_count < size_categories["small"]["threshold"]:
            size_categories["small"]["models"].append({"model": model, "parameters": param_count})
        elif param_count < size_categories["medium"]["threshold"]:
            size_categories["medium"]["models"].append({"model": model, "parameters": param_count})
        elif param_count < size_categories["large"]["threshold"]:
            size_categories["large"]["models"].append({"model": model, "parameters": param_count})
        else:
            size_categories["xlarge"]["models"].append({"model": model, "parameters": param_count})
    
    # Add statistics for each category
    for category, data in size_categories.items():
        models = data["models"]
        data["count"] = len(models)
        data["total_parameters"] = sum(m["parameters"] for m in models)
        data["avg_parameters"] = data["total_parameters"] / len(models) if models else 0
    
    return size_categories

def _generate_comprehensive_ess_analysis(model_ess_scores: Dict[str, float], 
                                       framework_ess: Dict[str, List[float]],
                                       parameter_estimator: ParameterEstimator,
                                       individual_runs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive ESS analysis for research"""
    
    if not model_ess_scores:
        return {"ess_calculation_disabled": True}
    
    # Use parameter estimator's comparison functionality
    ess_comparison = parameter_estimator.compare_ess_scores(model_ess_scores)
    
    # Framework-specific ESS analysis
    framework_ess_analysis = {}
    for framework, ess_scores in framework_ess.items():
        if ess_scores:
            framework_ess_analysis[framework] = {
                "average_ess": sum(ess_scores) / len(ess_scores),
                "best_ess": max(ess_scores),
                "worst_ess": min(ess_scores),
                "ess_range": max(ess_scores) - min(ess_scores),
                "model_count": len(ess_scores),
                "efficiency_distribution": {
                    "excellent": sum(1 for ess in ess_scores if ess >= 1.0),
                    "efficient": sum(1 for ess in ess_scores if 0.5 <= ess < 1.0),
                    "moderate": sum(1 for ess in ess_scores if 0.1 <= ess < 0.5),
                    "poor": sum(1 for ess in ess_scores if ess < 0.1)
                }
            }
    
    # ESS vs other metrics correlation
    correlations = _calculate_ess_correlations(model_ess_scores, individual_runs)
    
    # ESS improvement opportunities
    improvement_opportunities = _identify_ess_improvement_opportunities(
        model_ess_scores, individual_runs, parameter_estimator
    )
    
    return {
        **ess_comparison,
        "framework_ess_analysis": framework_ess_analysis,
        "ess_correlations": correlations,
        "improvement_opportunities": improvement_opportunities,
        "ess_research_insights": _generate_ess_research_insights(
            ess_comparison, framework_ess_analysis, correlations
        )
    }


def _calculate_ess_distribution(model_ess_scores: Dict[str, float], 
                                       framework_ess: Dict[str, List[float]],
                                       parameter_estimator: ParameterEstimator,
                                       individual_runs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive ESS analysis for research"""
    
    if not model_ess_scores:
        return {"ess_calculation_disabled": True}
    
    # Use parameter estimator's comparison functionality
    ess_comparison = parameter_estimator.compare_ess_scores(model_ess_scores)
    
    # Framework-specific ESS analysis
    framework_ess_analysis = {}
    for framework, ess_scores in framework_ess.items():
        if ess_scores:
            framework_ess_analysis[framework] = {
                "average_ess": sum(ess_scores) / len(ess_scores),
                "best_ess": max(ess_scores),
                "worst_ess": min(ess_scores),
                "ess_range": max(ess_scores) - min(ess_scores),
                "model_count": len(ess_scores),
                "efficiency_distribution": {
                    "excellent": sum(1 for ess in ess_scores if ess >= 1.0),
                    "efficient": sum(1 for ess in ess_scores if 0.5 <= ess < 1.0),
                    "moderate": sum(1 for ess in ess_scores if 0.1 <= ess < 0.5),
                    "poor": sum(1 for ess in ess_scores if ess < 0.1)
                }
            }
    
    # ESS vs other metrics correlation
    correlations = _calculate_ess_correlations(model_ess_scores, individual_runs)
    
    # ESS improvement opportunities
    improvement_opportunities = _identify_ess_improvement_opportunities(
        model_ess_scores, individual_runs, parameter_estimator
    )
    
    return {
        **ess_comparison,
        "framework_ess_analysis": framework_ess_analysis,
        "ess_correlations": correlations,
        "improvement_opportunities": improvement_opportunities,
        "ess_research_insights": _generate_ess_research_insights(
            ess_comparison, framework_ess_analysis, correlations
        )
    }


def _calculate_ess_correlations(model_ess_scores: Dict[str, float], 
                              individual_runs: Dict[str, Any]) -> Dict[str, float]:
    """Calculate correlations between ESS and other metrics"""
    import numpy as np
    
    correlations = {}
    
    # Gather data for correlation analysis
    ess_values = []
    co2_values = []
    accuracy_values = []
    water_values = []
    
    for run_id, result in individual_runs.items():
        model_name = result.get("model_name")
        if model_name in model_ess_scores:
            ess_values.append(model_ess_scores[model_name])
            co2_values.append(result.get("emissions_analysis", {}).get("total_kg_co2", 0))
            water_values.append(result.get("water_analysis", {}).get("total_water_liters", 0))
            
            # Get accuracy metric
            accuracy_metrics = result.get("accuracy_metrics", {})
            accuracy = (accuracy_metrics.get("accuracy") or 
                       accuracy_metrics.get("bleu_score") or 
                       accuracy_metrics.get("r2_score") or 0)
            accuracy_values.append(accuracy)
    
    # Calculate correlations
    if len(ess_values) > 1:
        try:
            correlations["ess_vs_co2"] = np.corrcoef(ess_values, co2_values)[0, 1]
            correlations["ess_vs_accuracy"] = np.corrcoef(ess_values, accuracy_values)[0, 1]
            correlations["ess_vs_water"] = np.corrcoef(ess_values, water_values)[0, 1]
        except:
            pass  # Handle cases with constant values
    
    return correlations


def _identify_ess_improvement_opportunities(model_ess_scores: Dict[str, float],
                                          individual_runs: Dict[str, Any],
                                          parameter_estimator: ParameterEstimator) -> Dict[str, Any]:
    """Identify opportunities for ESS improvement"""
    
    opportunities = {
        "low_efficiency_models": [],
        "quantization_candidates": [],
        "architecture_suggestions": [],
        "framework_recommendations": {}
    }
    
    # Identify low efficiency models
    for model_name, ess_score in model_ess_scores.items():
        if ess_score < 0.1:  # Poor ESS threshold
            # Find the corresponding run
            model_run = None
            for run_id, result in individual_runs.items():
                if result.get("model_name") == model_name:
                    model_run = result
                    break
            
            if model_run:
                param_count = model_run.get("model_analysis", {}).get("parameter_count", 0)
                co2_kg = model_run.get("emissions_analysis", {}).get("total_kg_co2", 0)
                
                opportunities["low_efficiency_models"].append({
                    "model": model_name,
                    "ess_score": ess_score,
                    "parameters": param_count,
                    "co2_kg": co2_kg,
                    "category": parameter_estimator.categorize_ess(ess_score)
                })
        
        # Identify quantization candidates (large models with moderate ESS)
        model_run = None
        for run_id, result in individual_runs.items():
            if result.get("model_name") == model_name:
                model_run = result
                break
        
        if model_run:
            param_count = model_run.get("model_analysis", {}).get("parameter_count", 0)
            if param_count > 1_000_000_000 and 0.1 <= ess_score < 0.5:  # Large model, moderate ESS
                opportunities["quantization_candidates"].append({
                    "model": model_name,
                    "parameters": param_count,
                    "ess_score": ess_score,
                    "potential_improvement": "High (quantization could improve ESS by 30-50%)"
                })
    
    # Framework-specific recommendations
    framework_ess = {}
    for run_id, result in individual_runs.items():
        framework = result.get("framework", "unknown")
        model_name = result.get("model_name")
        
        if model_name in model_ess_scores:
            if framework not in framework_ess:
                framework_ess[framework] = []
            framework_ess[framework].append(model_ess_scores[model_name])
    
    for framework, ess_scores in framework_ess.items():
        if ess_scores:
            avg_ess = sum(ess_scores) / len(ess_scores)
            if avg_ess < 0.2:
                opportunities["framework_recommendations"][framework] = "Consider model optimization or alternative architectures"
            elif avg_ess > 0.5:
                opportunities["framework_recommendations"][framework] = "Good ESS performance - maintain current practices"
    
    return opportunities


def _generate_ess_research_insights(ess_comparison: Dict[str, Any],
                                  framework_ess_analysis: Dict[str, Any],
                                  correlations: Dict[str, float]) -> List[str]:
    """Generate research insights about ESS patterns"""
    insights = []
    
    # Model ranking insights
    if "summary" in ess_comparison:
        summary = ess_comparison["summary"]
        best_model = summary.get("best_model", {})
        worst_model = summary.get("worst_model", {})
        
        if best_model and worst_model:
            improvement_factor = best_model["ess"] / worst_model["ess"] if worst_model["ess"] > 0 else float('inf')
            insights.append(f"ESS varies by {improvement_factor:.1f}x across models ({best_model['name']} vs {worst_model['name']})")
    
    # Framework insights
    if framework_ess_analysis:
        framework_avgs = {fw: data["average_ess"] for fw, data in framework_ess_analysis.items()}
        best_framework = max(framework_avgs.items(), key=lambda x: x[1])
        worst_framework = min(framework_avgs.items(), key=lambda x: x[1])
        
        insights.append(f"Most efficient framework: {best_framework[0]} (avg ESS: {best_framework[1]:.3f})")
        
        if best_framework[1] / worst_framework[1] > 2:
            insights.append(f"Significant framework efficiency gap: {best_framework[0]} vs {worst_framework[0]}")
    
    # Correlation insights
    if correlations:
        ess_co2_corr = correlations.get("ess_vs_co2", 0)
        ess_accuracy_corr = correlations.get("ess_vs_accuracy", 0)
        
        if abs(ess_co2_corr) > 0.7:
            insights.append(f"Strong ESS-CO₂ correlation ({ess_co2_corr:.3f}) - parameter efficiency directly impacts emissions")
        
        if ess_accuracy_corr > 0.5:
            insights.append("Higher ESS models tend to have better accuracy - efficient architectures perform well")
        elif ess_accuracy_corr < -0.5:
            insights.append("ESS-accuracy tradeoff observed - larger models may sacrifice efficiency for performance")
    
    return insights


def analyze_research_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze comprehensive research results and provide insights including water footprint
    
    Args:
        results: Results dictionary from run_comprehensive_analysis
        
    Returns:
        Dict with detailed analysis and recommendations including water efficiency
    """
    if "aggregated_analysis" not in results:
        return {"error": "No aggregated analysis found in results"}
    
    analysis = results["aggregated_analysis"]
    
    # Extract key metrics
    total_emissions = analysis["overall_statistics"]["total_co2_emissions_kg"]
    avg_emissions = analysis["overall_statistics"]["average_co2_per_run_kg"]
    avg_efficiency = analysis["overall_statistics"]["average_efficiency_score"]
    
    # Water footprint metrics
    total_water = analysis["overall_statistics"]["total_water_liters"]
    avg_water = analysis["overall_statistics"]["average_water_per_run_liters"]
    avg_water_efficiency = analysis["overall_statistics"]["average_water_efficiency_score"]

    # Extract ESS analysis
    ess_analysis = analysis.get("ess_analysis", {})
    parameter_analysis = analysis.get("parameter_analysis", {})
        
    
    # Generate recommendations
    recommendations = []
    
    if avg_emissions > 0.01:  # High emissions threshold
        recommendations.append("Consider model optimization or quantization to reduce carbon footprint")
    
    if avg_water > 1.0:  # High water consumption threshold
        recommendations.append("High water consumption detected - consider model quantization or edge deployment")
    
    if avg_efficiency < 0.5:  # Low efficiency threshold
        recommendations.append("Investigate hardware optimization or alternative deployment strategies")
    
    if avg_water_efficiency < 0.5:  # Low water efficiency threshold
        recommendations.append("Poor water efficiency - consider model compression or renewable energy sources")
    
    # Model comparison
    model_analysis = analysis.get("model_analysis", {})
    if len(model_analysis) > 1:
        # Best CO2 model
        best_co2_model = min(model_analysis.keys(), 
                            key=lambda m: model_analysis[m]["avg_co2_per_run_kg"])
        recommendations.append(f"Consider using {best_co2_model} for lower carbon impact")
        
        # Best water model
        best_water_model = min(model_analysis.keys(), 
                              key=lambda m: model_analysis[m]["avg_water_per_run_liters"])
        recommendations.append(f"Consider using {best_water_model} for lower water footprint")
    
    # Water-specific insights
    water_analysis = analysis.get("water_footprint_analysis", {})
    water_insights = []
    
    if total_water > 5.0:
        water_insights.append(f"High total water consumption: {total_water:.3f} L ({total_water/0.5:.1f} bottles)")
    
    improvement_potential = water_analysis.get("water_efficiency_improvement_potential", {})
    if improvement_potential.get("improvement_potential_percent", 0) > 20:
        water_insights.append(f"Significant water efficiency improvement possible: {improvement_potential['improvement_potential_percent']:.1f}%")
    
    return {
        "summary": {
            "total_emissions_kg": total_emissions,
            "average_emissions_per_run_kg": avg_emissions,
            "total_water_liters": total_water,
            "average_water_per_run_liters": avg_water,
            "water_bottles_equivalent": total_water / 0.5,
            "average_efficiency_score": avg_efficiency,
            "average_water_efficiency_score": avg_water_efficiency,
            "total_parameters": parameter_analysis.get("total_parameters_analyzed", 0),
            "average_parameters": parameter_analysis.get("average_parameters_per_model", 0),
            "average_ess_score": ess_analysis.get("summary", {}).get("average_ess", 0),
            "best_ess_model": ess_analysis.get("summary", {}).get("best_model", {}),
            "ess_enabled": not ess_analysis.get("ess_calculation_disabled", False),          
            "total_runs": analysis["overall_statistics"]["total_successful_runs"]
        },
        "recommendations": recommendations,
        "insights": analysis.get("research_insights", []),
        "water_insights": water_insights,
        "model_ranking_co2": sorted(
            model_analysis.items(),
            key=lambda x: x[1]["avg_co2_per_run_kg"]
        ) if model_analysis else [],
        "model_ranking_water": sorted(
            model_analysis.items(),
            key=lambda x: x[1]["avg_water_per_run_liters"]
        ) if model_analysis else [],
        "water_efficiency_potential": improvement_potential
    }

def _generate_ess_optimization_recommendations(ess_analysis: Dict[str, Any]) -> List[str]:
    """Generate ESS-focused optimization recommendations"""
    recommendations = []
    
    if ess_analysis.get("ess_calculation_disabled"):
        return ["Enable ESS calculation to get efficiency recommendations"]
    
    summary = ess_analysis.get("summary", {})
    avg_ess = summary.get("average_ess", 0)
    
    if avg_ess < 0.1:
        recommendations.append("Critical: Overall ESS is very low - urgent need for model optimization")
        recommendations.append("Consider: Model pruning, quantization, or architecture changes")
    elif avg_ess < 0.3:
        recommendations.append("Moderate ESS scores - room for significant efficiency improvements")
        recommendations.append("Investigate: Quantization and model compression techniques")
    elif avg_ess > 0.7:
        recommendations.append("Excellent ESS performance - maintain current efficient practices")
    
    # Framework-specific recommendations
    framework_analysis = ess_analysis.get("framework_ess_analysis", {})
    if framework_analysis:
        best_framework = max(framework_analysis.items(), key=lambda x: x[1]["average_ess"])
        recommendations.append(f"Most efficient framework: {best_framework[0]} (avg ESS: {best_framework[1]['average_ess']:.3f})")
    
    return recommendations


def _generate_parameter_insights(parameter_analysis: Dict[str, Any]) -> List[str]:
    """Generate parameter-focused insights"""
    insights = []
    
    total_params = parameter_analysis.get("total_parameters_analyzed", 0)
    if total_params > 0:
        insights.append(f"Analyzed {total_params:,} total parameters across all models")
        
        size_categories = parameter_analysis.get("parameter_distribution", {}).get("size_categories", {})
        for category, data in size_categories.items():
            if data["count"] > 0:
                insights.append(f"{category.capitalize()} models: {data['count']} ({data['total_parameters']:,} params)")
    
    sources = parameter_analysis.get("parameter_distribution", {}).get("parameter_sources", {})
    if sources:
        most_common_source = max(sources.items(), key=lambda x: x[1])
        insights.append(f"Parameter estimation: {most_common_source[1]} models used {most_common_source[0]} method")
    
    return insights