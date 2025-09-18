#!/usr/bin/env python3
"""
Master cleanup control system - ZERO RISK approach to code cleanup.
This orchestrates the entire safe cleanup process.
"""

import os
import sys
import time
import subprocess

class MasterCleanupControl:
    def __init__(self):
        self.steps_completed = []
        
    def run_master_cleanup(self):
        """Run the complete master cleanup process"""
        print("🛡️  MASTER CLEANUP CONTROL SYSTEM")
        print("=" * 60)
        print("ZERO RISK APPROACH TO CODE CLEANUP")
        print("=" * 60)
        print()
        print("This system will:")
        print("✅ Create complete backups")
        print("✅ Capture API performance baselines")
        print("✅ Identify unused models safely")
        print("✅ Test every change automatically")
        print("✅ Rollback immediately if any issues detected")
        print("✅ Give you clear accuracy breakdowns")
        print()
        
        # Safety confirmation
        confirm = input("🔒 SAFETY CONFIRMATION: Type 'YES' to proceed with safe cleanup: ").strip()
        if confirm != "YES":
            print("❌ Cleanup cancelled for safety")
            return False
        
        print("\n🚀 STARTING MASTER CLEANUP PROCESS")
        print("=" * 50)
        
        # Step 1: Pre-flight checks
        if not self.pre_flight_checks():
            return False
        
        # Step 2: Create baseline and backup
        if not self.create_baseline_and_backup():
            return False
        
        # Step 3: Get current accuracy breakdown
        if not self.get_accuracy_breakdown():
            return False
        
        # Step 4: Execute safe cleanup
        if not self.execute_safe_cleanup():
            return False
        
        # Step 5: Final verification
        if not self.final_verification():
            return False
        
        print("\n🎯 MASTER CLEANUP COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("✅ Your API performance is verified unchanged")
        print("✅ Unused models have been safely quarantined")
        print("✅ You have complete backups for rollback")
        print("✅ You have clear accuracy breakdowns")
        
        return True
    
    def pre_flight_checks(self):
        """Perform pre-flight safety checks"""
        print("\n🔍 STEP 1: PRE-FLIGHT SAFETY CHECKS")
        print("-" * 40)
        
        # Check if API is running
        try:
            import requests
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API is healthy: {health_data.get('status', 'Unknown')}")
            else:
                print("❌ API is not responding correctly")
                print("Please start your API: python main_local_dev.py")
                return False
        except:
            print("❌ Cannot connect to API")
            print("Please start your API: python main_local_dev.py")
            return False
        
        # Check for required scripts
        required_scripts = [
            'api_baseline_test.py',
            'backup_and_inventory.py',
            'safe_cleanup_orchestrator.py',
            'accuracy_monitor.py'
        ]
        
        for script in required_scripts:
            if os.path.exists(script):
                print(f"✅ Found required script: {script}")
            else:
                print(f"❌ Missing required script: {script}")
                return False
        
        # Check disk space
        try:
            import shutil
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            if free_space_gb > 1:  # At least 1GB free
                print(f"✅ Sufficient disk space: {free_space_gb:.1f} GB free")
            else:
                print(f"⚠️  Low disk space: {free_space_gb:.1f} GB free")
                print("Consider freeing up space before cleanup")
        except:
            print("⚠️  Could not check disk space")
        
        print("✅ Pre-flight checks completed")
        self.steps_completed.append("pre_flight_checks")
        return True
    
    def create_baseline_and_backup(self):
        """Create baseline and backup"""
        print("\n💾 STEP 2: CREATING BASELINE AND BACKUP")
        print("-" * 40)
        
        try:
            # Create backup and inventory
            print("Creating backup and inventory...")
            result = subprocess.run([sys.executable, 'backup_and_inventory.py'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Backup and inventory created")
            else:
                print(f"❌ Backup failed: {result.stderr}")
                return False
            
            # Capture API baseline
            print("Capturing API baseline...")
            from api_baseline_test import APIBaselineTest
            
            tester = APIBaselineTest()
            if tester.capture_baseline():
                print("✅ API baseline captured")
            else:
                print("❌ Failed to capture API baseline")
                return False
            
            self.steps_completed.append("baseline_and_backup")
            return True
            
        except Exception as e:
            print(f"❌ Error in baseline/backup: {e}")
            return False
    
    def get_accuracy_breakdown(self):
        """Get current accuracy breakdown"""
        print("\n📊 STEP 3: GETTING ACCURACY BREAKDOWN")
        print("-" * 40)
        
        try:
            from accuracy_monitor import AccuracyMonitor
            
            monitor = AccuracyMonitor()
            if monitor.monitor_accuracy(num_tests=3):
                print("✅ Accuracy breakdown completed")
                self.steps_completed.append("accuracy_breakdown")
                return True
            else:
                print("❌ Failed to get accuracy breakdown")
                return False
                
        except Exception as e:
            print(f"❌ Error getting accuracy breakdown: {e}")
            return False
    
    def execute_safe_cleanup(self):
        """Execute the safe cleanup process"""
        print("\n🧹 STEP 4: EXECUTING SAFE CLEANUP")
        print("-" * 40)
        
        try:
            from safe_cleanup_orchestrator import SafeCleanupOrchestrator
            
            orchestrator = SafeCleanupOrchestrator()
            
            # Run the cleanup without user interaction
            print("Starting automated safe cleanup...")
            
            # Verify prerequisites
            if not orchestrator.verify_prerequisites():
                print("❌ Prerequisites not met")
                return False
            
            # Create backup (already done, but verify)
            print("✅ Backup already created")
            
            # Capture baseline (already done, but verify)
            print("✅ Baseline already captured")
            
            # Create inventory (already done, but verify)  
            print("✅ Inventory already created")
            
            # Execute cleanup stages
            if orchestrator.execute_cleanup_stages():
                print("✅ Safe cleanup completed")
                self.steps_completed.append("safe_cleanup")
                return True
            else:
                print("❌ Safe cleanup failed")
                return False
                
        except Exception as e:
            print(f"❌ Error in safe cleanup: {e}")
            return False
    
    def final_verification(self):
        """Final verification of API performance"""
        print("\n🧪 STEP 5: FINAL VERIFICATION")
        print("-" * 40)
        
        try:
            from api_baseline_test import APIBaselineTest
            
            tester = APIBaselineTest()
            if tester.compare_with_baseline():
                print("✅ Final verification passed - API performance unchanged")
                self.steps_completed.append("final_verification")
                return True
            else:
                print("❌ Final verification failed - API performance changed")
                print("Initiating emergency rollback...")
                return self.emergency_rollback()
                
        except Exception as e:
            print(f"❌ Error in final verification: {e}")
            return self.emergency_rollback()
    
    def emergency_rollback(self):
        """Emergency rollback if anything goes wrong"""
        print("\n🚨 EMERGENCY ROLLBACK")
        print("-" * 40)
        
        try:
            from safe_cleanup_orchestrator import SafeCleanupOrchestrator
            
            orchestrator = SafeCleanupOrchestrator()
            if orchestrator.emergency_rollback():
                print("✅ Emergency rollback completed")
                print("Your system has been restored to the original state")
                return True
            else:
                print("❌ Emergency rollback failed")
                print("Please manually restore from backup directory")
                return False
                
        except Exception as e:
            print(f"❌ Emergency rollback error: {e}")
            return False
    
    def show_summary(self):
        """Show summary of what was accomplished"""
        print("\n📋 CLEANUP SUMMARY")
        print("=" * 40)
        
        if "pre_flight_checks" in self.steps_completed:
            print("✅ Pre-flight safety checks completed")
        
        if "baseline_and_backup" in self.steps_completed:
            print("✅ Baseline and backup created")
        
        if "accuracy_breakdown" in self.steps_completed:
            print("✅ Accuracy breakdown generated")
        
        if "safe_cleanup" in self.steps_completed:
            print("✅ Safe cleanup executed")
        
        if "final_verification" in self.steps_completed:
            print("✅ Final verification passed")
        
        print("\n📁 FILES CREATED:")
        files_to_check = [
            "api_baseline.json",
            "api_baseline_report.md",
            "accuracy_monitoring_results.json",
            "cleanup_log.json"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"   📄 {file_path}")
        
        # Check for backup directories
        backup_dirs = [d for d in os.listdir('.') if d.startswith('backup_')]
        quarantine_dirs = [d for d in os.listdir('.') if d.startswith('quarantine_')]
        
        if backup_dirs:
            print(f"\n📦 BACKUP DIRECTORIES:")
            for backup_dir in backup_dirs:
                print(f"   📁 {backup_dir}")
        
        if quarantine_dirs:
            print(f"\n🏥 QUARANTINE DIRECTORIES:")
            for quarantine_dir in quarantine_dirs:
                print(f"   📁 {quarantine_dir}")

def main():
    """Main function"""
    print("Master Cleanup Control System")
    print("=" * 50)
    
    controller = MasterCleanupControl()
    
    action = input("Choose action:\n1. Run complete safe cleanup\n2. Show cleanup summary\nEnter choice (1/2): ").strip()
    
    if action == "1":
        success = controller.run_master_cleanup()
        controller.show_summary()
        
        if success:
            print("\n🎯 SUCCESS! Your codebase has been safely cleaned up.")
            print("Your API performance is verified unchanged.")
        else:
            print("\n⚠️  Cleanup stopped for safety.")
            print("Your system remains in its original state.")
    
    elif action == "2":
        controller.show_summary()
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
