"""
Optimized Colab Setup (Python version)
Alternative to bash script for better error handling
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, shell=False):
    """Run command and print output."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd if shell else cmd.split(),
        capture_output=True,
        text=True,
        shell=shell
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def mount_drive():
    """Mount Google Drive."""
    print("📁 Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        
        # Create project folder
        project_dir = Path("/content/drive/MyDrive/model-dqn-snn")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Drive mounted and folder ready")
        return True
    except Exception as e:
        print(f"❌ Drive mount failed: {e}")
        return False


def setup_repository():
    """Clone or update repository."""
    print("\n📦 Setting up repository...")
    
    repo_dir = Path("/content/model-dqn-snn")
    
    if repo_dir.exists():
        print("   Repository exists, pulling latest changes...")
        os.chdir(repo_dir)
        run_command("git pull")
    else:
        print("   Cloning repository...")
        os.chdir("/content")
        run_command("git clone https://github.com/saoxdxd2/model-dqn-snn.git")
        os.chdir(repo_dir)
    
    return repo_dir


def install_dependencies():
    """Install Python dependencies efficiently."""
    print("\n📦 Installing dependencies...")
    
    # Use pip with optimizations
    run_command("pip install -q --no-cache-dir -r requirements.txt")
    
    print("✅ Dependencies installed")


def restore_checkpoint():
    """Restore checkpoint from Google Drive if exists."""
    print("\n🔄 Checking for saved checkpoint...")
    
    drive_checkpoint = Path("/content/drive/MyDrive/model-dqn-snn/encoding_checkpoint.pt")
    local_checkpoint = Path("datasets/vision_unified/encoding_checkpoint.pt")
    
    if drive_checkpoint.exists():
        local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_checkpoint, local_checkpoint)
        print("✅ Restored checkpoint from Drive")
        print(f"   Location: {local_checkpoint}")
        
        # Check checkpoint info
        import torch
        try:
            ckpt = torch.load(local_checkpoint, map_location='cpu')
            batch = ckpt.get('batch_count', 0)
            total = ckpt.get('total_batches', 0)
            progress = (batch / total * 100) if total > 0 else 0
            print(f"   Progress: {batch}/{total} batches ({progress:.1f}%)")
        except:
            pass
    else:
        print("ℹ️  No previous checkpoint found - starting fresh")


def main():
    """Main setup routine."""
    print("=" * 70)
    print("  Colab Setup: TRM Training Pipeline")
    print("=" * 70)
    
    # Step 1: Mount Drive
    if not mount_drive():
        print("⚠️  Continuing without Drive (no persistence)")
    
    # Step 2: Setup repository
    repo_dir = setup_repository()
    
    # Step 3: Install dependencies
    install_dependencies()
    
    # Step 4: Restore checkpoint
    restore_checkpoint()
    
    # Step 5: Start training
    print("\n" + "=" * 70)
    print("🚀 Starting training pipeline...")
    print("   Auto-saves every 2000 batches (~1 hour)")
    print("   Checkpoints sync to Google Drive automatically")
    print("=" * 70)
    print()
    
    # Run training
    result = subprocess.run([sys.executable, "train.py"])
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
