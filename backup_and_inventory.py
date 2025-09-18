#!/usr/bin/env python3
"""
Backup and inventory system for safe cleanup operations.
"""

import shutil
import os
import json
import time
from pathlib import Path
import hashlib

class BackupAndInventory:
    def __init__(self):
        self.backup_dir = f"backup_{int(time.time())}"
        self.inventory = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'backup_directory': self.backup_dir,
            'files_backed_up': [],
            'model_files': [],
            'api_files': [],
            'unused_files': [],
            'critical_files': []
        }
    
    def create_full_backup(self):
        """Create complete backup of current system"""
        print("💾 CREATING FULL SYSTEM BACKUP")
        print("=" * 50)
        
        # Create backup directory
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Critical files to always backup
        critical_files = [
            'main_local_dev.py',
            'main.py',
            'main_final_piece_classifier.py'
        ]
        
        # Model directories to backup
        model_dirs = [
            'models/',
            'runs/',
            'checkpoints/'
        ]
        
        # Backup critical files
        for file_path in critical_files:
            if os.path.exists(file_path):
                self.backup_file(file_path)
                self.inventory['critical_files'].append(file_path)
        
        # Backup model directories
        for dir_path in model_dirs:
            if os.path.exists(dir_path):
                self.backup_directory(dir_path)
        
        # Save inventory
        self.save_inventory()
        
        print(f"\n✅ BACKUP COMPLETE: {self.backup_dir}")
        return self.backup_dir
    
    def backup_file(self, file_path):
        """Backup a single file"""
        if os.path.exists(file_path):
            backup_path = os.path.join(self.backup_dir, file_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(file_path, backup_path)
            
            # Calculate hash for integrity
            file_hash = self.calculate_file_hash(file_path)
            
            self.inventory['files_backed_up'].append({
                'original_path': file_path,
                'backup_path': backup_path,
                'hash': file_hash,
                'size': os.path.getsize(file_path)
            })
            
            print(f"  📁 Backed up: {file_path}")
    
    def backup_directory(self, dir_path):
        """Backup entire directory"""
        if os.path.exists(dir_path):
            backup_path = os.path.join(self.backup_dir, dir_path)
            shutil.copytree(dir_path, backup_path, dirs_exist_ok=True)
            print(f"  📁 Backed up directory: {dir_path}")
    
    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def inventory_models(self):
        """Create comprehensive inventory of all model files"""
        print("\n🔍 INVENTORYING MODEL FILES")
        print("=" * 50)
        
        model_extensions = ['.pt', '.pth', '.pkl', '.h5', '.pb', '.onnx']
        
        # Search for model files
        for root, dirs, files in os.walk('.'):
            for file in files:
                if any(file.endswith(ext) for ext in model_extensions):
                    file_path = os.path.join(root, file)
                    file_info = self.analyze_model_file(file_path)
                    self.inventory['model_files'].append(file_info)
                    print(f"  🤖 Found model: {file_path} ({file_info['size_mb']:.1f} MB)")
        
        print(f"\n📊 Total models found: {len(self.inventory['model_files'])}")
    
    def analyze_model_file(self, file_path):
        """Analyze a model file"""
        stat = os.stat(file_path)
        return {
            'path': file_path,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified_time': time.ctime(stat.st_mtime),
            'hash': self.calculate_file_hash(file_path),
            'in_use': self.check_if_model_in_use(file_path)
        }
    
    def check_if_model_in_use(self, model_path):
        """Check if model file is referenced in API code"""
        api_files = [
            'main_local_dev.py',
            'main.py',
            'main_final_piece_classifier.py'
        ]
        
        model_name = os.path.basename(model_path)
        model_name_no_ext = os.path.splitext(model_name)[0]
        
        for api_file in api_files:
            if os.path.exists(api_file):
                with open(api_file, 'r') as f:
                    content = f.read()
                    if model_name in content or model_name_no_ext in content or model_path in content:
                        return {'used_in': api_file, 'status': 'IN_USE'}
        
        return {'used_in': None, 'status': 'POTENTIALLY_UNUSED'}
    
    def identify_unused_files(self):
        """Identify potentially unused files"""
        print("\n🗑️  IDENTIFYING UNUSED FILES")
        print("=" * 50)
        
        unused_count = 0
        for model_info in self.inventory['model_files']:
            if model_info['in_use']['status'] == 'POTENTIALLY_UNUSED':
                self.inventory['unused_files'].append(model_info)
                unused_count += 1
                print(f"  ❓ Potentially unused: {model_info['path']} ({model_info['size_mb']:.1f} MB)")
        
        if unused_count == 0:
            print("  ✅ No obviously unused files found")
        else:
            total_unused_mb = sum(f['size_mb'] for f in self.inventory['unused_files'])
            print(f"\n📊 Potentially unused: {unused_count} files ({total_unused_mb:.1f} MB)")
    
    def save_inventory(self):
        """Save inventory to file"""
        inventory_file = f"inventory_{int(time.time())}.json"
        with open(inventory_file, 'w') as f:
            json.dump(self.inventory, f, indent=2, default=str)
        
        print(f"\n📋 Inventory saved to: {inventory_file}")
        return inventory_file
    
    def generate_cleanup_plan(self):
        """Generate safe cleanup plan"""
        print("\n📋 GENERATING CLEANUP PLAN")
        print("=" * 50)
        
        plan = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'backup_directory': self.backup_dir,
            'steps': []
        }
        
        # Step 1: Backup verification
        plan['steps'].append({
            'step': 1,
            'action': 'verify_backup',
            'description': 'Verify backup integrity',
            'risk': 'LOW',
            'rollback': 'N/A'
        })
        
        # Step 2: API baseline test
        plan['steps'].append({
            'step': 2,
            'action': 'baseline_test',
            'description': 'Capture API baseline performance',
            'risk': 'NONE',
            'rollback': 'N/A'
        })
        
        # Step 3: Move unused files (don't delete)
        if self.inventory['unused_files']:
            plan['steps'].append({
                'step': 3,
                'action': 'move_unused_files',
                'description': f'Move {len(self.inventory["unused_files"])} potentially unused files to quarantine',
                'risk': 'LOW',
                'rollback': 'Move files back from quarantine'
            })
        
        # Step 4: Test API after each change
        plan['steps'].append({
            'step': 4,
            'action': 'test_after_changes',
            'description': 'Test API performance after each change',
            'risk': 'NONE',
            'rollback': 'Restore from backup if issues detected'
        })
        
        # Save plan
        plan_file = f"cleanup_plan_{int(time.time())}.json"
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"📋 Cleanup plan saved to: {plan_file}")
        return plan_file
    
    def create_quarantine_directory(self):
        """Create quarantine directory for unused files"""
        quarantine_dir = f"quarantine_{int(time.time())}"
        os.makedirs(quarantine_dir, exist_ok=True)
        print(f"🏥 Created quarantine directory: {quarantine_dir}")
        return quarantine_dir

def main():
    """Main function"""
    print("Backup and Inventory System")
    print("=" * 50)
    print("This will create a complete backup and inventory of your system")
    print("for safe cleanup operations.")
    print()
    
    backup_system = BackupAndInventory()
    
    # Create full backup
    backup_dir = backup_system.create_full_backup()
    
    # Inventory models
    backup_system.inventory_models()
    
    # Identify unused files
    backup_system.identify_unused_files()
    
    # Save inventory
    inventory_file = backup_system.save_inventory()
    
    # Generate cleanup plan
    plan_file = backup_system.generate_cleanup_plan()
    
    print("\n🎯 BACKUP AND INVENTORY COMPLETE!")
    print(f"Backup directory: {backup_dir}")
    print(f"Inventory file: {inventory_file}")
    print(f"Cleanup plan: {plan_file}")
    print("\nNext steps:")
    print("1. Run: python api_baseline_test.py (choose option 1)")
    print("2. Review the inventory and cleanup plan")
    print("3. Execute cleanup in safe stages")

if __name__ == "__main__":
    main()
