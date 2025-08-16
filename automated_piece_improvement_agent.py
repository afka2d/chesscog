#!/usr/bin/env python3
"""
Automated Piece Classification Improvement Agent

This agent runs in the background to improve piece classification accuracy
by implementing a multi-phase approach:

Phase 1: Dataset Expansion (Synthetic Data Generation)
Phase 2: Enhanced Model Training
Phase 3: Model Evaluation and Selection
Phase 4: Performance Analysis

Run this script and leave your computer unattended for several hours.
The agent will automatically improve your piece classification model.
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('piece_improvement_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PieceClassificationAgent:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.results_dir = self.base_dir / "automated_improvement_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.start_time = datetime.now()
        self.current_phase = 0
        self.total_phases = 4
        
        # Performance tracking
        self.initial_accuracy = None
        self.final_accuracy = None
        self.improvement_log = []
        
    def log_improvement(self, phase, description, accuracy=None, duration=None):
        """Log improvement progress"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'description': description,
            'accuracy': accuracy,
            'duration': duration
        }
        self.improvement_log.append(entry)
        
        # Save to file
        with open(self.results_dir / "improvement_log.json", 'w') as f:
            json.dump(self.improvement_log, f, indent=2)
    
    def run_command(self, command, description, timeout=3600):
        """Run a command with logging and timeout"""
        logger.info(f"🔄 {description}")
        logger.info(f"Command: {command}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.base_dir
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully in {duration:.1f}s")
                return True, result.stdout, duration
            else:
                logger.error(f"❌ {description} failed")
                logger.error(f"Error: {result.stderr}")
                return False, result.stderr, duration
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {description} timed out after {timeout}s")
            return False, "Timeout", timeout
        except Exception as e:
            logger.error(f"💥 {description} failed with exception: {e}")
            return False, str(e), time.time() - start_time
    
    def get_current_accuracy(self):
        """Get current model accuracy"""
        logger.info("📊 Measuring current model accuracy...")
        
        # Run the accuracy test script
        success, output, duration = self.run_command(
            "source venv/bin/activate && python test_uniform_model_accuracy.py",
            "Testing current model accuracy",
            timeout=300
        )
        
        if success:
            # Parse accuracy from output
            for line in output.split('\n'):
                if "Overall Accuracy:" in line:
                    accuracy = float(line.split(':')[1].strip().split()[0])
                    logger.info(f"Current accuracy: {accuracy:.4f}")
                    return accuracy
        
        logger.warning("Could not determine current accuracy")
        return None
    
    def phase_1_dataset_expansion(self):
        """Phase 1: Expand dataset with synthetic data"""
        logger.info("🚀 Starting Phase 1: Dataset Expansion")
        self.current_phase = 1
        
        # Step 1: Analyze current dataset
        logger.info("📈 Analyzing current dataset...")
        success, output, duration = self.run_command(
            "source venv/bin/activate && python expand_chess_dataset.py --analyze-only",
            "Analyzing current dataset",
            timeout=600
        )
        
        # Step 2: Generate synthetic data
        logger.info("🎨 Generating synthetic chess piece data...")
        success, output, duration = self.run_command(
            "source venv/bin/activate && python expand_chess_dataset.py --generate-synthetic",
            "Generating synthetic data",
            timeout=1800  # 30 minutes
        )
        
        if success:
            self.log_improvement(1, "Synthetic data generation completed", duration=duration)
        
        # Step 3: Merge datasets
        logger.info("🔗 Merging original and synthetic datasets...")
        success, output, duration = self.run_command(
            "source venv/bin/activate && python expand_chess_dataset.py --merge-datasets",
            "Merging datasets",
            timeout=900  # 15 minutes
        )
        
        if success:
            self.log_improvement(1, "Dataset merging completed", duration=duration)
        
        logger.info("✅ Phase 1 completed: Dataset expanded")
    
    def phase_2_enhanced_training(self):
        """Phase 2: Train enhanced models"""
        logger.info("🚀 Starting Phase 2: Enhanced Model Training")
        self.current_phase = 2
        
        # Step 1: Train ResNet50 with enhanced augmentation
        logger.info("🧠 Training ResNet50 with enhanced augmentation...")
        success, output, duration = self.run_command(
            "source venv/bin/activate && python enhanced_piece_classifier.py --model ResNet50 --epochs 20",
            "Training ResNet50 model",
            timeout=7200  # 2 hours
        )
        
        if success:
            self.log_improvement(2, "ResNet50 training completed", duration=duration)
        
        # Step 2: Train ensemble model
        logger.info("🎯 Training ensemble model...")
        success, output, duration = self.run_command(
            "source venv/bin/activate && python enhanced_piece_classifier.py --model Ensemble --epochs 15",
            "Training ensemble model",
            timeout=5400  # 1.5 hours
        )
        
        if success:
            self.log_improvement(2, "Ensemble model training completed", duration=duration)
        
        # Step 3: Train with focal loss
        logger.info("🎯 Training with focal loss...")
        success, output, duration = self.run_command(
            "source venv/bin/activate && python enhanced_piece_classifier.py --model ResNet50 --loss focal --epochs 15",
            "Training with focal loss",
            timeout=5400  # 1.5 hours
        )
        
        if success:
            self.log_improvement(2, "Focal loss training completed", duration=duration)
        
        logger.info("✅ Phase 2 completed: Enhanced models trained")
    
    def phase_3_model_evaluation(self):
        """Phase 3: Evaluate and select best model"""
        logger.info("🚀 Starting Phase 3: Model Evaluation")
        self.current_phase = 3
        
        # Step 1: Evaluate all models
        logger.info("📊 Evaluating all trained models...")
        success, output, duration = self.run_command(
            "source venv/bin/activate && python enhanced_piece_classifier.py --evaluate-all",
            "Evaluating all models",
            timeout=1800  # 30 minutes
        )
        
        if success:
            self.log_improvement(3, "Model evaluation completed", duration=duration)
        
        # Step 2: Select best model
        logger.info("🏆 Selecting best performing model...")
        success, output, duration = self.run_command(
            "source venv/bin/activate && python enhanced_piece_classifier.py --select-best",
            "Selecting best model",
            timeout=300  # 5 minutes
        )
        
        if success:
            self.log_improvement(3, "Best model selected", duration=duration)
        
        logger.info("✅ Phase 3 completed: Best model selected")
    
    def phase_4_performance_analysis(self):
        """Phase 4: Final performance analysis"""
        logger.info("🚀 Starting Phase 4: Performance Analysis")
        self.current_phase = 4
        
        # Step 1: Test final model accuracy
        logger.info("📊 Testing final model accuracy...")
        self.final_accuracy = self.get_current_accuracy()
        
        if self.final_accuracy:
            self.log_improvement(4, "Final accuracy measurement", accuracy=self.final_accuracy)
        
        # Step 2: Generate performance report
        logger.info("📋 Generating performance report...")
        success, output, duration = self.run_command(
            "source venv/bin/activate && python enhanced_piece_classifier.py --generate-report",
            "Generating performance report",
            timeout=600  # 10 minutes
        )
        
        if success:
            self.log_improvement(4, "Performance report generated", duration=duration)
        
        # Step 3: Create summary
        self.create_summary()
        
        logger.info("✅ Phase 4 completed: Performance analysis finished")
    
    def create_summary(self):
        """Create a summary of the improvement process"""
        total_duration = datetime.now() - self.start_time
        
        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_hours': total_duration.total_seconds() / 3600,
            'initial_accuracy': self.initial_accuracy,
            'final_accuracy': self.final_accuracy,
            'accuracy_improvement': self.final_accuracy - self.initial_accuracy if self.initial_accuracy and self.final_accuracy else None,
            'phases_completed': self.current_phase,
            'improvement_log': self.improvement_log
        }
        
        # Save summary
        with open(self.results_dir / "improvement_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("🎉 IMPROVEMENT PROCESS COMPLETED!")
        logger.info(f"⏱️  Total duration: {total_duration.total_seconds() / 3600:.1f} hours")
        logger.info(f"📊 Initial accuracy: {self.initial_accuracy:.4f}" if self.initial_accuracy else "📊 Initial accuracy: Unknown")
        logger.info(f"📊 Final accuracy: {self.final_accuracy:.4f}" if self.final_accuracy else "📊 Final accuracy: Unknown")
        
        if self.initial_accuracy and self.final_accuracy:
            improvement = self.final_accuracy - self.initial_accuracy
            logger.info(f"📈 Accuracy improvement: {improvement:.4f} ({improvement*100:.2f}%)")
        
        logger.info(f"📁 Results saved to: {self.results_dir}")
    
    def run(self):
        """Run the complete improvement process"""
        logger.info("🤖 Starting Automated Piece Classification Improvement Agent")
        logger.info(f"📁 Working directory: {self.base_dir}")
        logger.info(f"📁 Results directory: {self.results_dir}")
        
        # Get initial accuracy
        self.initial_accuracy = self.get_current_accuracy()
        if self.initial_accuracy:
            self.log_improvement(0, "Initial accuracy measurement", accuracy=self.initial_accuracy)
        
        try:
            # Run all phases
            self.phase_1_dataset_expansion()
            self.phase_2_enhanced_training()
            self.phase_3_model_evaluation()
            self.phase_4_performance_analysis()
            
            logger.info("🎉 All phases completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("⚠️ Process interrupted by user")
            self.create_summary()
        except Exception as e:
            logger.error(f"💥 Process failed with error: {e}")
            self.create_summary()
            raise

def main():
    """Main function"""
    print("🤖 Automated Piece Classification Improvement Agent")
    print("=" * 60)
    print("This agent will run for several hours to improve your piece classification.")
    print("You can leave your computer unattended during this process.")
    print()
    print("Expected improvements:")
    print("- Dataset expansion with synthetic data")
    print("- Enhanced model training (ResNet50, ensemble, focal loss)")
    print("- Model evaluation and selection")
    print("- Performance analysis and reporting")
    print()
    print("Estimated time: 4-6 hours")
    print()
    
    # Check if required files exist
    required_files = [
        "expand_chess_dataset.py",
        "enhanced_piece_classifier.py",
        "test_uniform_model_accuracy.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print()
        print("Please ensure all required files are present before running the agent.")
        return
    
    # Confirm before starting
    response = input("Do you want to start the improvement process? (y/N): ")
    if response.lower() != 'y':
        print("Process cancelled.")
        return
    
    # Create and run agent
    agent = PieceClassificationAgent()
    agent.run()

if __name__ == "__main__":
    main() 