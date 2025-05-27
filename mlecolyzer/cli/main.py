"""
Command Line Interface for ML-EcoLyzer

This module provides the main CLI interface for running ML-EcoLyzer environmental impact analyses.
"""

import click
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .. import EcoLyzer, run_comprehensive_analysis, get_info
from ..utils.helpers import (
    load_config_from_file, get_default_config, print_banner, 
    get_system_info, setup_logging
)
from ..utils.validation import validate_config, validate_research_config
from ..monitoring.hardware import detect_hardware_capabilities


@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def cli(verbose: bool, quiet: bool):
    """ML-EcoLyzer: Machine Learning Environmental Impact Analysis Framework"""
    if verbose:
        setup_logging("DEBUG")
    elif quiet:
        setup_logging("ERROR")
    else:
        setup_logging("INFO")


@cli.command()
@click.option('--model', '-m', required=True, help='Model name (HuggingFace identifier)')
@click.option('--dataset', '-d', required=True, help='Dataset name (HuggingFace identifier)')
@click.option('--task', '-t', default='text', help='Task type (text, image, audio)')
@click.option('--project', '-p', help='Project name for tracking')
@click.option('--limit', '-l', type=int, help='Limit number of samples for quick testing')
@click.option('--output', '-o', help='Output directory')
@click.option('--wandb/--no-wandb', default=None, help='Enable/disable wandb tracking')
@click.option('--gpu/--no-gpu', default=None, help='Force GPU usage on/off')
def analyze(model: str, dataset: str, task: str, project: Optional[str], 
            limit: Optional[int], output: Optional[str], wandb: Optional[bool],
            gpu: Optional[bool]):
    """Run environmental impact analysis on a single model-dataset pair"""
    
    print_banner("ML-EcoLyzer Environmental Analysis", f"Model: {model} | Dataset: {dataset}")
    
    # Create configuration
    config = {
        "project": project or f"analysis_{model.replace('/', '_')}_{dataset.replace('/', '_')}",
        "models": [{"name": model, "task": task}],
        "datasets": [{"name": dataset, "task": task}],
    }
    
    # Add optional parameters
    if limit:
        config["datasets"][0]["limit"] = limit
    if output:
        config["output_dir"] = output
    if wandb is not None:
        config["enable_wandb"] = wandb
    
    try:
        # Validate configuration
        validate_config(config)
        
        # Run analysis
        analyzer = EcoLyzer(config)
        results = analyzer.run()
        
        # Print summary
        _print_analysis_summary(results)
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True), 
              help='Configuration file (YAML or JSON)')
@click.option('--output', '-o', help='Output directory (overrides config)')
@click.option('--dry-run', is_flag=True, help='Validate configuration without running')
def run(config: str, output: Optional[str], dry_run: bool):
    """Run environmental analysis with configuration file"""
    
    try:
        # Load configuration
        config_dict = load_config_from_file(config)
        
        # Override output directory if specified
        if output:
            config_dict["output_dir"] = output
        
        # Validate configuration
        validate_config(config_dict)
        
        if dry_run:
            click.echo("✅ Configuration is valid")
            _print_config_summary(config_dict)
            return
        
        print_banner("ML-EcoLyzer Analysis", f"Config: {config}")
        
        # Run analysis
        analyzer = EcoLyzer(config_dict)
        results = analyzer.run()
        
        # Print summary
        _print_analysis_summary(results)
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Research configuration file (YAML or JSON)')
@click.option('--output', '-o', help='Output directory (overrides config)')
@click.option('--dry-run', is_flag=True, help='Validate configuration without running')
def research(config: str, output: Optional[str], dry_run: bool):
    """Run comprehensive environmental impact research study"""
    
    try:
        # Load configuration
        config_dict = load_config_from_file(config)
        
        # Override output directory if specified
        if output:
            config_dict["output_dir"] = output
        
        # Validate research configuration
        validate_research_config(config_dict)
        
        if dry_run:
            click.echo("✅ Research configuration is valid")
            _print_research_summary(config_dict)
            return
        
        print_banner("ML-EcoLyzer Comprehensive Research", f"Config: {config}")
        
        # Run research
        results = run_comprehensive_analysis(config_dict)
        
        # Print summary
        _print_research_results_summary(results)
        
    except Exception as e:
        click.echo(f"❌ Research failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show system information and environmental monitoring capabilities"""
    
    print_banner("ML-EcoLyzer System Information")
    
    # Get package info
    package_info = get_info()
    
    click.echo("📦 Package Information:")
    click.echo(f"   Version: {package_info['version']}")
    click.echo(f"   Framework: ML-EcoLyzer")
    click.echo(f"   License: {package_info['license']}")
    
    # Get hardware capabilities
    capabilities = detect_hardware_capabilities()
    
    click.echo("\n🖥️ Hardware Capabilities:")
    click.echo(f"   Platform: {capabilities.platform}")
    click.echo(f"   Device Category: {capabilities.device_category}")
    click.echo(f"   GPU Available: {capabilities.has_gpu}")
    if capabilities.has_gpu:
        click.echo(f"   GPU Count: {capabilities.gpu_count}")
        for i, gpu_name in enumerate(capabilities.gpu_names):
            click.echo(f"   GPU {i}: {gpu_name}")
    click.echo(f"   Battery: {capabilities.has_battery}")
    click.echo(f"   Monitoring Methods: {', '.join(capabilities.monitoring_methods)}")
    
    # Get system info
    system_info = get_system_info()
    
    click.echo("\n💻 System Information:")
    click.echo(f"   OS: {system_info['platform']['system']} {system_info['platform']['release']}")
    click.echo(f"   Architecture: {system_info['platform']['machine']}")
    click.echo(f"   Python: {system_info['python']['version'].split()[0]}")
    
    if 'memory' in system_info:
        total_gb = system_info['memory']['total'] / (1024**3)
        available_gb = system_info['memory']['available'] / (1024**3)
        click.echo(f"   Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
    
    if 'cpu' in system_info:
        click.echo(f"   CPU Cores: {system_info['cpu']['logical_cores']} logical")
    
    click.echo("\n🌱 Environmental Monitoring:")
    click.echo(f"   Carbon Tracking: ✅ CodeCarbon integration")
    click.echo(f"   Power Monitoring: {'✅' if 'power' in capabilities.monitoring_methods else '⚠️  Limited'}")
    click.echo(f"   Thermal Monitoring: {'✅' if capabilities.thermal_sensors else '⚠️  Basic'}")
    click.echo(f"   Quantization Analysis: ✅ Available")


@cli.command()
@click.option('--output', '-o', default='ml_ecolyzer_config.yaml', 
              help='Output configuration file')
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), default='yaml',
              help='Configuration format')
@click.option('--template', '-t', type=click.Choice(['basic', 'research', 'quantization', 'multimodal']),
              default='basic', help='Configuration template')
def init(output: str, format: str, template: str):
    """Generate configuration template for environmental analysis"""
    
    config = _generate_config_template(template)
    
    try:
        from ..utils.helpers import save_config_to_file
        save_config_to_file(config, output, format)
        click.echo(f"✅ Configuration template saved to: {output}")
        click.echo(f"📝 Edit the file and run: mlecolyzer run -c {output}")
        
    except Exception as e:
        click.echo(f"❌ Failed to save configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Configuration file to validate')
@click.option('--research', is_flag=True, help='Validate as research configuration')
def validate(config: str, research: bool):
    """Validate configuration file for environmental analysis"""
    
    try:
        # Load configuration
        config_dict = load_config_from_file(config)
        
        # Validate
        if research:
            validate_research_config(config_dict)
            click.echo("✅ Research configuration is valid")
            _print_research_summary(config_dict)
        else:
            validate_config(config_dict)
            click.echo("✅ Configuration is valid")
            _print_config_summary(config_dict)
            
    except Exception as e:
        click.echo(f"❌ Configuration validation failed: {e}", err=True)
        sys.exit(1)


def _print_analysis_summary(results: Dict[str, Any]):
    """Print environmental analysis results summary"""
    
    if "final_report" in results:
        report = results["final_report"]
        summary = report.get("analysis_summary", {})
        
        click.echo("\n🌱 Environmental Impact Analysis Summary:")
        click.echo(f"   Total Evaluations: {summary.get('total_evaluations', 0)}")
        click.echo(f"   Failed Evaluations: {summary.get('failed_evaluations', 0)}")
        click.echo(f"   Total CO2 Emissions: {summary.get('total_co2_emissions_kg', 0):.6f} kg")
        click.echo(f"   Average CO2 per Evaluation: {summary.get('average_co2_per_evaluation_kg', 0):.6f} kg")
        click.echo(f"   Hardware Category: {summary.get('hardware_category', 'unknown')}")
        
        # Environmental analysis
        env_analysis = report.get("environmental_analysis", {})
        click.echo(f"   Overall Efficiency Score: {env_analysis.get('overall_efficiency_score', 0):.3f}")
    
    # Individual results
    individual_results = [k for k in results.keys() if not k.startswith('ERROR') and k != 'final_report']
    
    if individual_results:
        click.echo(f"\n✅ Successfully analyzed {len(individual_results)} model-dataset combinations")
        
        for key in individual_results[:3]:  # Show first 3 results
            result = results[key]
            co2 = result.get("emissions_analysis", {}).get("total_kg_co2", 0)
            efficiency = result.get("environmental_assessment", {}).get("integrated_assessment", {}).get("overall_efficiency_score", 0)
            click.echo(f"   {result.get('model_name', 'Unknown')} on {result.get('dataset_name', 'Unknown')}: {co2:.6f} kg CO2, efficiency: {efficiency:.3f}")
        
        if len(individual_results) > 3:
            click.echo(f"   ... and {len(individual_results) - 3} more")


def _print_config_summary(config: Dict[str, Any]):
    """Print configuration summary"""
    
    click.echo("\n📋 Configuration Summary:")
    click.echo(f"   Project: {config.get('project', 'unknown')}")
    click.echo(f"   Models: {len(config.get('models', []))}")
    click.echo(f"   Datasets: {len(config.get('datasets', []))}")
    click.echo(f"   Total Combinations: {len(config.get('models', [])) * len(config.get('datasets', []))}")
    click.echo(f"   Output Directory: {config.get('output_dir', '.')}")
    click.echo(f"   Monitoring Duration: {config.get('monitoring_duration', 300)}s")
    click.echo(f"   Wandb Enabled: {config.get('enable_wandb', 'auto')}")


def _print_research_summary(config: Dict[str, Any]):
    """Print research configuration summary"""
    
    _print_config_summary(config)
    
    total_combinations = len(config.get('models', [])) * len(config.get('datasets', []))
    estimated_hours = (total_combinations * 5) / 60  # 5 minutes per combination
    
    click.echo(f"\n🔬 Research Study Scale:")
    click.echo(f"   Estimated Duration: {estimated_hours:.1f} hours")
    click.echo(f"   Individual Analyses: {total_combinations}")
    
    if total_combinations > 20:
        click.echo("   ⚡ Consider running in stages for large-scale studies")


def _print_research_results_summary(results: Dict[str, Any]):
    """Print research results summary"""
    
    metadata = results.get("research_metadata", {})
    aggregated = results.get("aggregated_analysis", {})
    
    click.echo("\n🎉 Research Study Complete!")
    click.echo(f"   Total Duration: {metadata.get('total_duration_seconds', 0) / 3600:.1f} hours")
    click.echo(f"   Successful Analyses: {metadata.get('successful_runs', 0)}")
    click.echo(f"   Failed Analyses: {metadata.get('failed_runs', 0)}")
    click.echo(f"   Success Rate: {metadata.get('success_rate', 0) * 100:.1f}%")
    
    if aggregated and "overall_statistics" in aggregated:
        stats = aggregated["overall_statistics"]
        click.echo(f"   Total CO2 Emissions: {stats.get('total_co2_emissions_kg', 0):.6f} kg")
        click.echo(f"   Average CO2 per Analysis: {stats.get('average_co2_per_run_kg', 0):.6f} kg")
        click.echo(f"   Average Efficiency: {stats.get('average_efficiency_score', 0):.3f}")
    
    # Research insights
    insights = aggregated.get("research_insights", [])
    if insights:
        click.echo("\n💡 Key Research Insights:")
        for insight in insights[:3]:
            click.echo(f"   • {insight}")


def _generate_config_template(template: str) -> Dict[str, Any]:
    """Generate configuration template based on type"""
    
    base_config = get_default_config()
    
    if template == "basic":
        return {
            **base_config,
            "project": "basic_environmental_analysis",
            "models": [
                {"name": "gpt2", "task": "text"},
                {"name": "distilbert-base-uncased", "task": "text"}
            ],
            "datasets": [
                {"name": "wikitext", "subset": "wikitext-2-raw-v1", "task": "text", "limit": 100},
                {"name": "imdb", "task": "text", "limit": 100}
            ]
        }
    
    elif template == "research":
        return {
            **base_config,
            "project": "comprehensive_ml_environmental_study",
            "models": [
                {"name": "gpt2", "task": "text"},
                {"name": "microsoft/DialoGPT-medium", "task": "text"},
                {"name": "facebook/bart-base", "task": "text"}
            ],
            "datasets": [
                {"name": "wikitext", "subset": "wikitext-2-raw-v1", "task": "text", "limit": 1000},
                {"name": "squad", "task": "text", "limit": 500},
                {"name": "imdb", "task": "text", "limit": 800}
            ],
            "enable_quantization_analysis": True,
            "monitoring_duration": 600
        }
    
    elif template == "quantization":
        return {
            **base_config,
            "project": "quantization_impact_study",
            "models": [
                {"name": "bert-base-uncased", "task": "text"},
                {"name": "bert-base-uncased", "task": "text", "quantization": {"enabled": True, "method": "dynamic"}},
                {"name": "bert-base-uncased", "task": "text", "quantization": {"enabled": True, "method": "static"}}
            ],
            "datasets": [
                {"name": "glue", "subset": "sst2", "task": "text", "limit": 500}
            ],
            "enable_quantization_analysis": True
        }
    
    elif template == "multimodal":
        return {
            **base_config,
            "project": "multimodal_environmental_analysis",
            "models": [
                {"name": "gpt2", "task": "text"},
                {"name": "microsoft/resnet-50", "task": "image"},
                {"name": "facebook/wav2vec2-base-960h", "task": "audio"}
            ],
            "datasets": [
                {"name": "wikitext", "subset": "wikitext-2-raw-v1", "task": "text", "limit": 100},
                {"name": "imagenet-1k", "task": "image", "limit": 100},
                {"name": "librispeech_asr", "task": "audio", "limit": 100}
            ]
        }
    
    return base_config


def main():
    """Main CLI entry point"""
    cli()


def research_main():
    """Research CLI entry point"""
    # This is for the mlecolyzer-research command
    @click.command()
    @click.option('--config', '-c', required=True, type=click.Path(exists=True),
                  help='Research configuration file (YAML or JSON)')
    @click.option('--output', '-o', help='Output directory (overrides config)')
    @click.option('--dry-run', is_flag=True, help='Validate configuration without running')
    def research_cmd(config: str, output: str, dry_run: bool):
        """Run comprehensive environmental impact research study"""
        
        try:
            # Load configuration
            config_dict = load_config_from_file(config)
            
            # Override output directory if specified
            if output:
                config_dict["output_dir"] = output
            
            # Validate research configuration
            validate_research_config(config_dict)
            
            if dry_run:
                click.echo("✅ Research configuration is valid")
                _print_research_summary(config_dict)
                return
            
            print_banner("ML-EcoLyzer Comprehensive Research", f"Config: {config}")
            
            # Run research
            results = run_comprehensive_analysis(config_dict)
            
            # Print summary
            _print_research_results_summary(results)
            
        except Exception as e:
            click.echo(f"❌ Research failed: {e}", err=True)
            sys.exit(1)
    
    research_cmd()


if __name__ == "__main__":
    main()