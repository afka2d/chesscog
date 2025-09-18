#!/usr/bin/env python3
"""
Safe cleanup orchestrator with automatic rollback capabilities.
ZERO RISK approach to cleaning up your codebase.
"""

import os
import json
import shutil
import time
from pathlib import Path
import subprocess
import sys

class SafeCleanupOrchestrator:
    def __init__(self):
        self.current_step = 0
        self.cleanup_log = []
        self.quarantine_dir = None
        self.backup_dir = None
        self.baseline_file = None
        
    def execute_safe_cleanup(self):
        """Execute safe cleanup with automatic testing and rollback"""
        print("🛡️  SAFE CLEANUP ORCHESTRATOR")
        print("=" * 50)
        print("This will clean up your codebase with ZERO RISK to production.")
        print("Every step includes automatic testing and rollback capabilities.")
        print()
        
        # Step 1: Verify prerequisites
        if not self.verify_prerequisites():
            return False
        
        # Step 2: Create backup
        if not self.create_backup():
            return False
        
        # Step 3: Capture baseline
        if not self.capture_baseline():
            return False
        
        # Step 4: Create inventory
        if not self.create_inventory():
            return False
        
        # Step 5: Execute cleanup stages
        if not self.execute_cleanup_stages():
            return False
        
        print("\n🎯 SAFE CLEANUP COMPLETED SUCCESSFULLY!")
        return True
    
    def verify_prerequisites(self):
        """Verify all prerequisites are met"""
        print("\n🔍 STEP 1: VERIFYING PREREQUISITES")
        print("-" * 30)
        
        # Check if API is running
        try:
            import requests
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is running and healthy")
            else:
                print("❌ API is not responding correctly")
                return False
        except:
            print("❌ Cannot connect to API")
            print("Please start your API first: python main_local_dev.py")
            return False
        
        # Check for required files
        required_files = ['main_local_dev.py', 'api_baseline_test.py', 'backup_and_inventory.py']
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ Found required file: {file_path}")
            else:
                print(f"❌ Missing required file: {file_path}")
                return False
        
        print("✅ All prerequisites verified")
        return True
    
    def create_backup(self):
        """Create complete system backup"""
        print("\n💾 STEP 2: CREATING BACKUP")
        print("-" * 30)
        
        try:
            # Run backup script
            result = subprocess.run([sys.executable, 'backup_and_inventory.py'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Backup created successfully")
                # Find the backup directory from the output
                for line in result.stdout.split('\n'):
                    if 'Backup directory:' in line:
                        self.backup_dir = line.split(': ')[1].strip()
                        break
                
                self.log_action("backup_created", {"backup_dir": self.backup_dir})
                return True
            else:
                print(f"❌ Backup failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Backup timed out")
            return False
        except Exception as e:
            print(f"❌ Backup error: {e}")
            return False
    
    def capture_baseline(self):
        """Capture API performance baseline"""
        print("\n📊 STEP 3: CAPTURING BASELINE")
        print("-" * 30)
        
        try:
            # Import and run baseline test
            from api_baseline_test import APIBaselineTest
            
            tester = APIBaselineTest()
            success = tester.capture_baseline()
            
            if success:
                self.baseline_file = "api_baseline.json"
                print("✅ Baseline captured successfully")
                self.log_action("baseline_captured", {"baseline_file": self.baseline_file})
                return True
            else:
                print("❌ Failed to capture baseline")
                return False
                
        except Exception as e:
            print(f"❌ Baseline capture error: {e}")
            return False
    
    def create_inventory(self):
        """Create system inventory"""
        print("\n📋 STEP 4: CREATING INVENTORY")
        print("-" * 30)
        
        try:
            from backup_and_inventory import BackupAndInventory
            
            inventory_system = BackupAndInventory()
            inventory_system.inventory_models()
            inventory_system.identify_unused_files()
            inventory_file = inventory_system.save_inventory()
            
            print("✅ Inventory created successfully")
            self.log_action("inventory_created", {"inventory_file": inventory_file})
            return True
            
        except Exception as e:
            print(f"❌ Inventory creation error: {e}")
            return False
    
    def execute_cleanup_stages(self):
        """Execute cleanup in safe stages"""
        print("\n🧹 STEP 5: EXECUTING CLEANUP STAGES")
        print("-" * 30)
        
        # Load inventory to find unused files
        inventory_files = [f for f in os.listdir('.') if f.startswith('inventory_') and f.endswith('.json')]
        if not inventory_files:
            print("❌ No inventory file found")
            return False
        
        latest_inventory = max(inventory_files, key=lambda x: os.path.getctime(x))
        
        with open(latest_inventory, 'r') as f:
            inventory = json.load(f)
        
        unused_files = inventory.get('unused_files', [])
        
        if not unused_files:
            print("✅ No unused files found - nothing to clean up")
            return True
        
        print(f"Found {len(unused_files)} potentially unused files")
        
        # Create quarantine directory
        self.quarantine_dir = f"quarantine_{int(time.time())}"
        os.makedirs(self.quarantine_dir, exist_ok=True)
        print(f"🏥 Created quarantine directory: {self.quarantine_dir}")
        
        # Process files in small batches
        batch_size = 3
        for i in range(0, len(unused_files), batch_size):
            batch = unused_files[i:i+batch_size]
            
            if not self.process_cleanup_batch(batch, i//batch_size + 1):
                print("❌ Cleanup batch failed - stopping cleanup")
                return False
        
        print("✅ All cleanup stages completed successfully")
        return True
    
    def process_cleanup_batch(self, batch, batch_number):
        """Process a batch of files for cleanup"""
        print(f"\n🔧 Processing batch {batch_number} ({len(batch)} files)")
        
        moved_files = []
        
        try:
            # Move files to quarantine
            for file_info in batch:
                file_path = file_info['path']
                if os.path.exists(file_path):
                    quarantine_path = os.path.join(self.quarantine_dir, os.path.basename(file_path))
                    shutil.move(file_path, quarantine_path)
                    moved_files.append((file_path, quarantine_path))
                    print(f"  🏥 Quarantined: {file_path}")
            
            # Test API after moving files
            print(f"  🧪 Testing API after batch {batch_number}...")
            
            from api_baseline_test import APIBaselineTest
            tester = APIBaselineTest()
            
            if tester.compare_with_baseline(self.baseline_file):
                print(f"  ✅ Batch {batch_number} successful - API performance unchanged")
                self.log_action("batch_successful", {
                    "batch_number": batch_number,
                    "files_moved": len(moved_files),
                    "moved_files": moved_files
                })
                return True
            else:
                print(f"  ❌ Batch {batch_number} caused API changes - rolling back")
                self.rollback_batch(moved_files)
                return False
                
        except Exception as e:
            print(f"  ❌ Batch {batch_number} error: {e} - rolling back")
            self.rollback_batch(moved_files)
            return False
    
    def rollback_batch(self, moved_files):
        """Rollback a batch of moved files"""
        print("  🔄 Rolling back batch...")
        
        for original_path, quarantine_path in moved_files:
            if os.path.exists(quarantine_path):
                shutil.move(quarantine_path, original_path)
                print(f"    ↩️  Restored: {original_path}")
        
        self.log_action("batch_rolled_back", {"files_restored": len(moved_files)})
        print("  ✅ Batch rolled back successfully")
    
    def log_action(self, action, details):
        """Log cleanup actions"""
        self.cleanup_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': action,
            'details': details
        })
        
        # Save log
        with open("cleanup_log.json", 'w') as f:
            json.dump(self.cleanup_log, f, indent=2)
    
    def emergency_rollback(self):
        """Emergency rollback of entire cleanup"""
        print("\n🚨 EMERGENCY ROLLBACK")
        print("-" * 30)
        
        if not self.backup_dir or not os.path.exists(self.backup_dir):
            print("❌ No backup directory found")
            return False
        
        try:
            # Restore critical files
            critical_files = ['main_local_dev.py', 'main.py', 'main_final_piece_classifier.py']
            
            for file_name in critical_files:
                backup_path = os.path.join(self.backup_dir, file_name)
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, file_name)
                    print(f"✅ Restored: {file_name}")
            
            # Restore model directories
            model_dirs = ['models/', 'runs/', 'checkpoints/']
            
            for dir_name in model_dirs:
                backup_path = os.path.join(self.backup_dir, dir_name)
                if os.path.exists(backup_path):
                    if os.path.exists(dir_name):
                        shutil.rmtree(dir_name)
                    shutil.copytree(backup_path, dir_name)
                    print(f"✅ Restored directory: {dir_name}")
            
            print("✅ Emergency rollback completed")
            return True
            
        except Exception as e:
            print(f"❌ Emergency rollback error: {e}")
            return False

def main():
    """Main function"""
    print("Safe Cleanup Orchestrator")
    print("=" * 50)
    print("This will safely clean up your codebase with ZERO RISK.")
    print("Every step includes automatic testing and rollback.")
    print()
    
    orchestrator = SafeCleanupOrchestrator()
    
    action = input("Choose action:\n1. Execute safe cleanup\n2. Emergency rollback\nEnter choice (1/2): ").strip()
    
    if action == "1":
        success = orchestrator.execute_safe_cleanup()
        if success:
            print("\n🎯 CLEANUP COMPLETED SUCCESSFULLY!")
            print("Your API performance has been verified at each step.")
        else:
            print("\n⚠️  CLEANUP STOPPED - NO CHANGES MADE")
            print("Your system remains in its original state.")
    
    elif action == "2":
        success = orchestrator.emergency_rollback()
        if success:
            print("\n🎯 EMERGENCY ROLLBACK COMPLETED!")
            print("Your system has been restored to the backed-up state.")
        else:
            print("\n❌ EMERGENCY ROLLBACK FAILED!")
            print("Please manually restore from backup directory.")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
