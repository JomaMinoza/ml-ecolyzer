"""
Water Footprint Utilities

This module provides utility functions for calculating, analyzing, and comparing
water footprints in machine learning workloads.
"""

from typing import Dict, List, Any, Optional, Tuple
import math


def calculate_water_equivalents(water_liters: float) -> Dict[str, float]:
    """
    Calculate water consumption in various practical equivalents
    
    Args:
        water_liters: Water consumption in liters
        
    Returns:
        Dictionary with various equivalent measurements
    """
    return {
        # Beverages
        "bottles_500ml": water_liters / 0.5,
        "bottles_1l": water_liters / 1.0,
        "glasses_250ml": water_liters / 0.25,
        "coffee_cups_240ml": water_liters / 0.24,
        "soda_cans_355ml": water_liters / 0.355,
        
        # Larger volumes
        "gallons_us": water_liters / 3.785,
        "gallons_imperial": water_liters / 4.546,
        "liters": water_liters,
        
        # Daily consumption references
        "daily_drinking_water": water_liters / 2.0,  # WHO recommendation: 2L/day
        "shower_minutes": water_liters / 9.5,  # Average 9.5L/minute shower
        "toilet_flushes": water_liters / 6.0,  # Average 6L per flush
        "dishwasher_loads": water_liters / 15.0,  # Average 15L per load
        
        # Perspective comparisons
        "bathtub_fill": water_liters / 300.0,  # Average bathtub: 300L
        "swimming_pool_olympic": water_liters / 2500000.0,  # Olympic pool: 2.5M L
    }


def compare_water_footprints(models_water: Dict[str, float], 
                           include_equivalents: bool = True) -> Dict[str, Any]:
    """
    Compare water footprints across multiple models
    
    Args:
        models_water: Dictionary mapping model names to water consumption (liters)
        include_equivalents: Whether to include practical equivalents
        
    Returns:
        Comprehensive comparison analysis
    """
    if not models_water:
        return {"error": "No models provided for comparison"}
    
    # Sort models by water consumption
    sorted_models = sorted(models_water.items(), key=lambda x: x[1])
    
    best_model = sorted_models[0]
    worst_model = sorted_models[-1]
    total_water = sum(models_water.values())
    avg_water = total_water / len(models_water)
    
    # Calculate relative improvements
    improvements = {}
    for model, water in models_water.items():
        if water > 0:
            improvement_vs_worst = ((worst_model[1] - water) / worst_model[1]) * 100
            improvement_vs_avg = ((avg_water - water) / avg_water) * 100
            improvements[model] = {
                "vs_worst_percent": improvement_vs_worst,
                "vs_average_percent": improvement_vs_avg,
                "absolute_difference_vs_worst_liters": worst_model[1] - water,
                "absolute_difference_vs_avg_liters": avg_water - water
            }
    
    comparison = {
        "summary": {
            "total_models": len(models_water),
            "total_water_liters": total_water,
            "average_water_liters": avg_water,
            "best_model": {"name": best_model[0], "water_liters": best_model[1]},
            "worst_model": {"name": worst_model[0], "water_liters": worst_model[1]},
            "range_liters": worst_model[1] - best_model[1],
            "coefficient_of_variation": (
                math.sqrt(sum((w - avg_water) ** 2 for w in models_water.values()) / len(models_water)) / avg_water
            ) if avg_water > 0 else 0
        },
        "rankings": [
            {"rank": i + 1, "model": model, "water_liters": water}
            for i, (model, water) in enumerate(sorted_models)
        ],
        "improvements": improvements
    }
    
    if include_equivalents:
        comparison["equivalents"] = {
            "total": calculate_water_equivalents(total_water),
            "average": calculate_water_equivalents(avg_water),
            "best_model": calculate_water_equivalents(best_model[1]),
            "worst_model": calculate_water_equivalents(worst_model[1])
        }
    
    return comparison


def estimate_water_savings_from_quantization(
    original_water: float,
    quantization_method: str = "dynamic",
    target_precision: str = "int8"
) -> Dict[str, float]:
    """
    Estimate water savings from model quantization
    
    Args:
        original_water: Original water consumption in liters
        quantization_method: Type of quantization (dynamic, static, qat)
        target_precision: Target precision (int8, int4, float16)
        
    Returns:
        Estimated water savings analysis
    """
    # Savings factors based on empirical observations
    savings_factors = {
        "dynamic": {
            "int8": 0.25,    # 25% savings
            "int4": 0.40,    # 40% savings
            "float16": 0.15  # 15% savings
        },
        "static": {
            "int8": 0.30,    # 30% savings
            "int4": 0.45,    # 45% savings
            "float16": 0.20  # 20% savings
        },
        "qat": {
            "int8": 0.35,    # 35% savings (best quality)
            "int4": 0.50,    # 50% savings
            "float16": 0.25  # 25% savings
        }
    }
    
    # Get savings factor
    base_savings = savings_factors.get(quantization_method, {}).get(target_precision, 0.2)
    
    # Apply additional factors
    # Cooling overhead reduction (quantization reduces heat → less cooling)
    cooling_savings_bonus = 0.05  # Additional 5% from reduced cooling
    total_savings_factor = min(base_savings + cooling_savings_bonus, 0.6)  # Cap at 60%
    
    water_saved = original_water * total_savings_factor
    new_water_consumption = original_water - water_saved
    
    return {
        "original_water_liters": original_water,
        "estimated_savings_factor": total_savings_factor,
        "estimated_savings_percent": total_savings_factor * 100,
        "water_saved_liters": water_saved,
        "new_water_consumption_liters": new_water_consumption,
        "quantization_method": quantization_method,
        "target_precision": target_precision,
        "water_bottles_saved": water_saved / 0.5,
        "cooling_savings_bonus": cooling_savings_bonus * 100,
        "equivalents_saved": calculate_water_equivalents(water_saved)
    }


def water_intensity_lookup(region: str) -> Dict[str, Any]:
    """
    Look up water intensity information for a region
    
    Args:
        region: Region identifier
        
    Returns:
        Water intensity information for the region
    """
    water_intensity_data = {
        # Very Low (1.0-1.5 L/kWh)
        "iceland": {
            "intensity_l_per_kwh": 1.2,
            "category": "very_low",
            "energy_mix": "Geothermal (70%), Hydro (30%)",
            "climate": "Cold, minimal cooling needed",
            "notes": "Abundant renewable energy, cold climate reduces cooling needs"
        },
        "norway": {
            "intensity_l_per_kwh": 1.4,
            "category": "very_low", 
            "energy_mix": "Hydro (95%), Wind (3%)",
            "climate": "Cold, minimal cooling needed",
            "notes": "Hydroelectric dominance, cold Nordic climate"
        },
        "costa_rica": {
            "intensity_l_per_kwh": 1.4,
            "category": "very_low",
            "energy_mix": "Hydro (70%), Renewables (25%)",
            "climate": "Tropical but renewable-heavy",
            "notes": "High renewable penetration despite warm climate"
        },
        
        # Low (1.5-2.5 L/kWh)
        "switzerland": {
            "intensity_l_per_kwh": 1.8,
            "category": "low",
            "energy_mix": "Hydro (56%), Nuclear (33%)",
            "climate": "Temperate, moderate cooling",
            "notes": "Hydroelectric and nuclear mix, Alpine cooling advantage"
        },
        "brazil": {
            "intensity_l_per_kwh": 2.0,
            "category": "low",
            "energy_mix": "Hydro (65%), Renewables (20%)",
            "climate": "Tropical but hydro-dominated",
            "notes": "Large hydroelectric capacity, some thermal plants"
        },
        "canada": {
            "intensity_l_per_kwh": 2.1,
            "category": "low",
            "energy_mix": "Hydro (60%), Nuclear (15%)",
            "climate": "Cold, low cooling requirements",
            "notes": "Abundant hydro resources, cold climate advantage"
        },
        "france": {
            "intensity_l_per_kwh": 2.2,
            "category": "low",
            "energy_mix": "Nuclear (70%), Renewables (20%)",
            "climate": "Temperate, moderate cooling",
            "notes": "Nuclear-dominant grid, moderate water usage"
        },
        
        # Medium (2.5-3.5 L/kWh)
        "uk": {
            "intensity_l_per_kwh": 2.5,
            "category": "medium",
            "energy_mix": "Gas (40%), Renewables (35%)",
            "climate": "Maritime, moderate cooling",
            "notes": "Transitioning to renewables, gas backup"
        },
        "germany": {
            "intensity_l_per_kwh": 2.8,
            "category": "medium",
            "energy_mix": "Renewables (45%), Gas (25%)",
            "climate": "Continental, moderate cooling",
            "notes": "Energiewende renewable transition ongoing"
        },
        "us_east": {
            "intensity_l_per_kwh": 3.0,
            "category": "medium",
            "energy_mix": "Gas (35%), Nuclear (20%), Coal (15%)",
            "climate": "Continental, variable cooling needs",
            "notes": "Mixed grid with significant nuclear, transitioning from coal"
        },
        "us_central": {
            "intensity_l_per_kwh": 3.0,
            "category": "medium",
            "energy_mix": "Gas (40%), Wind (25%), Coal (20%)",
            "climate": "Continental, hot summers",
            "notes": "Growing wind capacity, significant gas generation"
        },
        "japan": {
            "intensity_l_per_kwh": 3.1,
            "category": "medium",
            "energy_mix": "Gas (35%), Coal (30%), Nuclear (6%)",
            "climate": "Humid subtropical, high cooling needs",
            "notes": "Post-Fukushima nuclear reduction, fossil fuel dependence"
        },
        "gcp_europe": {
            "intensity_l_per_kwh": 2.2,
            "category": "low",
            "energy_mix": "Renewable-focused data centers",
            "climate": "Varies, efficient cooling",
            "notes": "Google's renewable energy commitments, efficient cooling"
        },
        
        # High (3.5-4.5 L/kWh)
        "south_korea": {
            "intensity_l_per_kwh": 3.3,
            "category": "high",
            "energy_mix": "Coal (40%), Gas (25%), Nuclear (25%)",
            "climate": "Continental, hot humid summers",
            "notes": "Coal-heavy grid, high cooling demands"
        },
        "us_southwest": {
            "intensity_l_per_kwh": 4.2,
            "category": "high",
            "energy_mix": "Gas (45%), Coal (25%), Solar (15%)",
            "climate": "Desert, very high cooling needs",
            "notes": "Extreme heat requires intensive cooling, water scarcity"
        },
        "us_texas": {
            "intensity_l_per_kwh": 3.8,
            "category": "high",
            "energy_mix": "Gas (45%), Coal (20%), Wind (20%)",
            "climate": "Hot, high cooling demands",
            "notes": "Hot climate, significant fossil fuel generation"
        },
        "china_north": {
            "intensity_l_per_kwh": 4.5,
            "category": "very_high",
            "energy_mix": "Coal (70%), Gas (15%)",
            "climate": "Continental, hot summers",
            "notes": "Coal-dominant grid, high water consumption plants"
        },
        "india": {
            "intensity_l_per_kwh": 3.9,
            "category": "high",
            "energy_mix": "Coal (60%), Renewables (25%)",
            "climate": "Tropical, very high cooling needs",
            "notes": "Coal-heavy grid, extreme heat requires significant cooling"
        },
        "australia": {
            "intensity_l_per_kwh": 4.1,
            "category": "high",
            "energy_mix": "Coal (50%), Gas (20%), Renewables (25%)",
            "climate": "Hot, dry, high cooling needs",
            "notes": "Coal dominance, hot climate, water scarcity issues"
        },
        
        # Very High (4.5+ L/kWh)
        "middle_east": {
            "intensity_l_per_kwh": 4.8,
            "category": "very_high",
            "energy_mix": "Oil (60%), Gas (35%)",
            "climate": "Desert, extreme cooling needs",
            "notes": "Oil/gas generation, extreme heat, water scarcity"
        },
        
        # Cloud provider optimized
        "aws_us_west": {
            "intensity_l_per_kwh": 3.5,
            "category": "high",
            "energy_mix": "Mixed grid, some renewables",
            "climate": "Hot, dry regions",
            "notes": "AWS West Coast regions, efficiency improvements"
        },
        "aws_us_east": {
            "intensity_l_per_kwh": 3.0,
            "category": "medium",
            "energy_mix": "Mixed grid, nuclear component",
            "climate": "Continental, moderate cooling",
            "notes": "AWS East Coast regions, grid diversity"
        },
        "aws_eu_west": {
            "intensity_l_per_kwh": 2.3,
            "category": "low",
            "energy_mix": "Renewable-heavy European grid",
            "climate": "Maritime, efficient cooling",
            "notes": "AWS European regions, renewable commitments"
        },
        "azure_europe": {
            "intensity_l_per_kwh": 2.4,
            "category": "low",
            "energy_mix": "Renewable commitments",
            "climate": "Varies, efficient cooling",
            "notes": "Microsoft carbon negative commitments"
        },
        "gcp_us": {
            "intensity_l_per_kwh": 3.2,
            "category": "medium",
            "energy_mix": "Mixed US grid, renewable investments",
            "climate": "Varies by region",
            "notes": "Google carbon neutral commitments, efficiency focus"
        },
        
        # Global default
        "global_average": {
            "intensity_l_per_kwh": 2.5,
            "category": "medium",
            "energy_mix": "Global average mix",
            "climate": "Various",
            "notes": "Worldwide average across all generation types"
        }
    }
    
    return water_intensity_data.get(region, water_intensity_data["global_average"])


def generate_water_efficiency_report(results: Dict[str, Any], 
                                   include_comparisons: bool = True) -> str:
    """
    Generate a comprehensive water efficiency report
    
    Args:
        results: ML-EcoLyzer results dictionary
        include_comparisons: Whether to include comparative analysis
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("🌊 WATER FOOTPRINT EFFICIENCY REPORT")
    report_lines.append("=" * 50)
    
    # Check if we have final report
    final_report = results.get("final_report", {})
    if not final_report:
        report_lines.append("❌ No final report found in results")
        return "\n".join(report_lines)
    
    # Analysis summary
    summary = final_report.get("analysis_summary", {})
    water_impact = final_report.get("water_impact_analysis", {})
    
    report_lines.append("\n📊 Overall Water Impact:")
    report_lines.append(f"   Total Water Consumption: {summary.get('total_water_liters', 0):.3f} L")
    report_lines.append(f"   Average per Model: {summary.get('average_water_per_evaluation_liters', 0):.3f} L")
    report_lines.append(f"   Bottle Equivalents: {summary.get('water_bottles_equivalent', 0):.1f} × 500ml")
    report_lines.append(f"   Gallon Equivalents: {water_impact.get('water_gallons_equivalent', 0):.2f} gallons")
    
    # Regional information
    report_lines.append(f"\n🌍 Regional Context:")
    region = summary.get('region', 'unknown')
    water_intensity = summary.get('water_intensity_factor', 0)
    region_info = water_intensity_lookup(region)
    
    report_lines.append(f"   Region: {region}")
    report_lines.append(f"   Water Intensity: {water_intensity:.2f} L/kWh ({region_info['category']})")
    report_lines.append(f"   Energy Mix: {region_info.get('energy_mix', 'Unknown')}")
    report_lines.append(f"   Climate: {region_info.get('climate', 'Unknown')}")
    
    # Framework comparison if available
    framework_analysis = final_report.get("framework_analysis", {})
    if framework_analysis and include_comparisons:
        report_lines.append(f"\n🔧 Framework Water Efficiency:")
        
        framework_water = {
            fw: stats["total_water_liters"] 
            for fw, stats in framework_analysis.items()
        }
        
        if framework_water:
            comparison = compare_water_footprints(framework_water, include_equivalents=False)
            best = comparison["summary"]["best_model"]
            worst = comparison["summary"]["worst_model"]
            
            report_lines.append(f"   Most Efficient: {best['name']} ({best['water_liters']:.3f} L)")
            report_lines.append(f"   Least Efficient: {worst['name']} ({worst['water_liters']:.3f} L)")
            
            if len(framework_water) > 1:
                improvement = ((worst['water_liters'] - best['water_liters']) / worst['water_liters']) * 100
                report_lines.append(f"   Efficiency Range: {improvement:.1f}% improvement possible")
    
    # Recommendations
    recommendations = water_impact.get("water_efficiency_recommendations", [])
    if recommendations:
        report_lines.append(f"\n💡 Water Efficiency Recommendations:")
        for rec in recommendations:
            report_lines.append(f"   • {rec}")
    
    # Individual model results (top 5 by water efficiency)
    individual_results = {
        k: v for k, v in results.items() 
        if not k.startswith('ERROR') and k != 'final_report'
    }
    
    if individual_results and include_comparisons:
        # Sort by water consumption
        sorted_results = sorted(
            individual_results.items(),
            key=lambda x: x[1].get("water_analysis", {}).get("total_water_liters", float('inf'))
        )
        
        report_lines.append(f"\n🏆 Top Water-Efficient Models:")
        for i, (key, result) in enumerate(sorted_results[:5]):
            model_name = result.get("model_name", "Unknown")
            water = result.get("water_analysis", {}).get("total_water_liters", 0)
            bottles = result.get("water_analysis", {}).get("water_equivalent_bottles", 0)
            efficiency = result.get("environmental_assessment", {}).get("water_analysis", {}).get("water_efficiency", 0)
            
            report_lines.append(f"   {i+1}. {model_name}")
            report_lines.append(f"      Water: {water:.3f} L ({bottles:.1f} bottles)")
            report_lines.append(f"      Efficiency Score: {efficiency:.3f}")
    
    # Water savings potential
    if "quantization_analysis" in final_report:
        quant = final_report["quantization_analysis"]
        potential_savings = quant.get("potential_water_savings_percent", 0)
        bottles_saved = quant.get("water_bottles_saved", 0)
        
        if potential_savings > 0:
            report_lines.append(f"\n⚖️ Quantization Water Savings Potential:")
            report_lines.append(f"   Estimated Savings: {potential_savings:.1f}%")
            report_lines.append(f"   Water Bottles Saved: {bottles_saved:.1f}")
            report_lines.append(f"   Methods: {', '.join(quant.get('quantization_methods', []))}")
    
    report_lines.append("\n" + "=" * 50)
    report_lines.append("Generated by ML-EcoLyzer Water Footprint Analysis")
    
    return "\n".join(report_lines)


def calculate_regional_water_impact_comparison() -> Dict[str, Any]:
    """
    Compare water impact across different regions for the same workload
    
    Returns:
        Regional comparison analysis
    """
    # Example: 1 kWh workload across regions
    base_energy_kwh = 1.0
    
    regions_to_compare = [
        "iceland", "norway", "france", "uk", "germany", 
        "us_east", "us_texas", "china_north", "india", "middle_east"
    ]
    
    regional_analysis = {}
    
    for region in regions_to_compare:
        region_info = water_intensity_lookup(region)
        water_consumption = base_energy_kwh * region_info["intensity_l_per_kwh"]
        
        regional_analysis[region] = {
            "water_liters": water_consumption,
            "water_bottles": water_consumption / 0.5,
            "intensity_factor": region_info["intensity_l_per_kwh"],
            "category": region_info["category"],
            "energy_mix": region_info["energy_mix"],
            "climate": region_info["climate"]
        }
    
    # Find best and worst regions
    sorted_regions = sorted(regional_analysis.items(), key=lambda x: x[1]["water_liters"])
    best_region = sorted_regions[0]
    worst_region = sorted_regions[-1]
    
    # Calculate potential improvement
    max_improvement = ((worst_region[1]["water_liters"] - best_region[1]["water_liters"]) / 
                      worst_region[1]["water_liters"]) * 100
    
    return {
        "comparison_basis": f"{base_energy_kwh} kWh workload",
        "best_region": {
            "name": best_region[0],
            "water_liters": best_region[1]["water_liters"],
            "category": best_region[1]["category"]
        },
        "worst_region": {
            "name": worst_region[0], 
            "water_liters": worst_region[1]["water_liters"],
            "category": worst_region[1]["category"]
        },
        "max_improvement_percent": max_improvement,
        "regional_details": regional_analysis,
        "summary": {
            "total_regions_analyzed": len(regions_to_compare),
            "water_range_liters": worst_region[1]["water_liters"] - best_region[1]["water_liters"],
            "average_water_liters": sum(r["water_liters"] for r in regional_analysis.values()) / len(regional_analysis)
        }
    }