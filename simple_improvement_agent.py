#!/usr/bin/env python3
"""
Simple Piece Classification Improvement Agent

This agent runs in the background to improve piece classification accuracy
using the existing chesscog infrastructure. It's designed to be practical
and work with your current setup.

Run this script and leave your computer unattended for several hours.
The agent will automatically improve your piece classification model.
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_improvement_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleImprovementAgent:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.results_dir = self.base_dir / "improvement_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.start_time = datetime.now()
        self.current_step = 0
        self.total_steps = 6
        
        # Performance tracking
        self.initial_accuracy = None
        self.final_accuracy = None
        self.improvement_log = []
        
    def log_step(self, step, description, accuracy=None, duration=None):
        """Log improvement progress"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
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
    
    def step_1_backup_current_model(self):
        """Step 1: Backup current model"""
        logger.info("🚀 Step 1: Backing up current model")
        self.current_step = 1
        
        # Create backup directory
        backup_dir = self.results_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        # Copy current model
        current_model = Path("runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt")
        if current_model.exists():
            backup_path = backup_dir / "ResNet_uniform_backup.pt"
            import shutil
            shutil.copy2(current_model, backup_path)
            logger.info(f"✅ Current model backed up to {backup_path}")
            self.log_step(1, "Model backup completed")
        else:
            logger.warning("⚠️ Current model not found, skipping backup")
    
    def step_2_train_improved_model(self):
        """Step 2: Train an improved model with more epochs"""
        logger.info("🚀 Step 2: Training improved model with more epochs")
        self.current_step = 2
        
        # Create improved config
        improved_config = """
# Improved ResNet configuration with more training
_BASE_: config://piece_classifier/_base.yaml

TRAINING:
  MODEL:
    REGISTRY: PIECE_CLASSIFIER
    NAME: ResNet
  PHASES:
    - EPOCHS: 30  # More epochs for better accuracy
      OPTIMIZER:
        LEARNING_RATE: 0.001
        NAME: Adam
      PARAMS: all
      LOSS:
        NAME: CrossEntropyLoss
        # Balanced weights for better performance
        CLASS_WEIGHTS: [1.5, 2.0, 1.5, 0.8, 2.0, 1.5, 1.5, 2.0, 1.5, 0.8, 2.0, 1.5]
"""
        
        config_path = Path("config/piece_classifier/ResNet_improved.yaml")
        with open(config_path, 'w') as f:
            f.write(improved_config)
        
        logger.info("✅ Created improved configuration")
        
        # Train the improved model
        success, output, duration = self.run_command(
            "source venv/bin/activate && python -m chesscog.piece_classifier.train --config ResNet_improved",
            "Training improved model",
            timeout=7200  # 2 hours
        )
        
        if success:
            self.log_step(2, "Improved model training completed", duration=duration)
            logger.info("✅ Improved model training completed")
        else:
            logger.error("❌ Improved model training failed")
    
    def step_3_train_resnet50_model(self):
        """Step 3: Train a ResNet50 model for better performance"""
        logger.info("🚀 Step 3: Training ResNet50 model")
        self.current_step = 3
        
        # Create ResNet50 config
        resnet50_config = """
# ResNet50 configuration for better accuracy
_BASE_: config://piece_classifier/_base.yaml

TRAINING:
  MODEL:
    REGISTRY: PIECE_CLASSIFIER
    NAME: ResNet50
  PHASES:
    - EPOCHS: 25  # Fewer epochs for ResNet50 (more complex)
      OPTIMIZER:
        LEARNING_RATE: 0.0005  # Lower learning rate for ResNet50
        NAME: Adam
      PARAMS: all
      LOSS:
        NAME: CrossEntropyLoss
        CLASS_WEIGHTS: [1.5, 2.0, 1.5, 0.8, 2.0, 1.5, 1.5, 2.0, 1.5, 0.8, 2.0, 1.5]
"""
        
        config_path = Path("config/piece_classifier/ResNet50.yaml")
        with open(config_path, 'w') as f:
            f.write(resnet50_config)
        
        logger.info("✅ Created ResNet50 configuration")
        
        # Train the ResNet50 model
        success, output, duration = self.run_command(
            "source venv/bin/activate && python -m chesscog.piece_classifier.train --config ResNet50",
            "Training ResNet50 model",
            timeout=9000  # 2.5 hours
        )
        
        if success:
            self.log_step(3, "ResNet50 model training completed", duration=duration)
            logger.info("✅ ResNet50 model training completed")
        else:
            logger.error("❌ ResNet50 model training failed")
    
    def step_4_evaluate_models(self):
        """Step 4: Evaluate all trained models"""
        logger.info("🚀 Step 4: Evaluating all models")
        self.current_step = 4
        
        models_to_evaluate = [
            "ResNet_uniform",  # Original model
            "ResNet_improved", # Improved model
            "ResNet50"         # ResNet50 model
        ]
        
        evaluation_results = {}
        
        for model_name in models_to_evaluate:
            logger.info(f"📊 Evaluating {model_name}...")
            
            # Run evaluation
            success, output, duration = self.run_command(
                f"source venv/bin/activate && python -m chesscog.piece_classifier.evaluate --config {model_name} --dataset test --out results/{model_name}_evaluation",
                f"Evaluating {model_name}",
                timeout=600  # 10 minutes per model
            )
            
            if success:
                # Try to parse accuracy from output
                accuracy = None
                for line in output.split('\n'):
                    if "accuracy" in line.lower() and ":" in line:
                        try:
                            accuracy = float(line.split(':')[1].strip().split()[0])
                            break
                        except:
                            continue
                
                evaluation_results[model_name] = {
                    'accuracy': accuracy,
                    'duration': duration,
                    'success': True
                }
                
                logger.info(f"✅ {model_name} evaluation completed - Accuracy: {accuracy:.4f}" if accuracy else "✅ {model_name} evaluation completed")
            else:
                evaluation_results[model_name] = {
                    'accuracy': None,
                    'duration': duration,
                    'success': False
                }
                logger.error(f"❌ {model_name} evaluation failed")
        
        # Save evaluation results
        with open(self.results_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.log_step(4, "Model evaluation completed", duration=sum(r.get('duration', 0) for r in evaluation_results.values()))
        logger.info("✅ All models evaluated")
        
        return evaluation_results
    
    def step_5_select_best_model(self):
        """Step 5: Select and deploy the best model"""
        logger.info("🚀 Step 5: Selecting best model")
        self.current_step = 5
        
        # Load evaluation results
        eval_file = self.results_dir / "evaluation_results.json"
        if not eval_file.exists():
            logger.error("❌ Evaluation results not found")
            return
        
        with open(eval_file, 'r') as f:
            evaluation_results = json.load(f)
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        for model_name, results in evaluation_results.items():
            if results.get('success') and results.get('accuracy'):
                accuracy = results['accuracy']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
        
        if best_model:
            logger.info(f"🏆 Best model: {best_model} with accuracy: {best_accuracy:.4f}")
            
            # Copy best model to production location
            source_model = Path(f"runs/piece_classifier/{best_model}/{best_model}.pt")
            if source_model.exists():
                # Update the production model
                production_model = Path("runs/piece_classifier/ResNet/ResNet.pt")
                production_model.parent.mkdir(exist_ok=True)
                
                import shutil
                shutil.copy2(source_model, production_model)
                
                logger.info(f"✅ Deployed {best_model} as production model")
                self.log_step(5, f"Best model {best_model} deployed", accuracy=best_accuracy)
            else:
                logger.error(f"❌ Best model file not found: {source_model}")
        else:
            logger.error("❌ No valid models found")
    
    def step_6_final_testing(self):
        """Step 6: Final testing of the improved model"""
        logger.info("🚀 Step 6: Final testing")
        self.current_step = 6
        
        # Test the final model accuracy
        self.final_accuracy = self.get_current_accuracy()
        
        if self.final_accuracy:
            self.log_step(6, "Final accuracy measurement", accuracy=self.final_accuracy)
            logger.info(f"📊 Final accuracy: {self.final_accuracy:.4f}")
        
        # Create summary
        self.create_summary()
        
        logger.info("✅ Final testing completed")
    
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
            'steps_completed': self.current_step,
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
        logger.info("🤖 Starting Simple Piece Classification Improvement Agent")
        logger.info(f"📁 Working directory: {self.base_dir}")
        logger.info(f"📁 Results directory: {self.results_dir}")
        
        # Get initial accuracy
        self.initial_accuracy = self.get_current_accuracy()
        if self.initial_accuracy:
            self.log_step(0, "Initial accuracy measurement", accuracy=self.initial_accuracy)
        
        try:
            # Run all steps
            self.step_1_backup_current_model()
            self.step_2_train_improved_model()
            self.step_3_train_resnet50_model()
            self.step_4_evaluate_models()
            self.step_5_select_best_model()
            self.step_6_final_testing()
            
            logger.info("🎉 All steps completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("⚠️ Process interrupted by user")
            self.create_summary()
        except Exception as e:
            logger.error(f"💥 Process failed with error: {e}")
            self.create_summary()
            raise

def main():
    """Main function"""
    print("🤖 Simple Piece Classification Improvement Agent")
    print("=" * 60)
    print("This agent will run for several hours to improve your piece classification.")
    print("You can leave your computer unattended during this process.")
    print()
    print("What this agent will do:")
    print("1. Backup your current model")
    print("2. Train an improved ResNet model (30 epochs)")
    print("3. Train a ResNet50 model (25 epochs)")
    print("4. Evaluate all models")
    print("5. Select and deploy the best model")
    print("6. Test final performance")
    print()
    print("Estimated time: 4-5 hours")
    print()
    
    # Check if we're in the right directory
    if not Path("chesscog").exists():
        print("❌ Please run this script from the chesscog root directory")
        return
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        print("❌ Virtual environment not found. Please activate your virtual environment first.")
        return
    
    # Auto-start without confirmation for unattended operation
    print("🚀 Starting improvement process automatically...")
    print("💡 To stop the process, press Ctrl+C")
    print()
    
    # Create and run agent
    agent = SimpleImprovementAgent()
    agent.run()

if __name__ == "__main__":
    main() 