"""
Google Drive Integration for Colab Session Persistence

Handles:
- Auto-mounting Google Drive
- Syncing datasets to persistent storage
- Resuming after 3h session timeout
"""

import os
import shutil
from pathlib import Path


def mount_gdrive():
    """Mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✅ Google Drive mounted at /content/drive")
        return True
    except ImportError:
        print("⚠️  Not running in Colab, skipping Drive mount")
        return False
    except Exception as e:
        print(f"❌ Drive mount failed: {e}")
        return False


def setup_persistent_storage(project_name="model-dqn-snn"):
    """
    Setup persistent storage on Google Drive.
    
    Returns:
        persistent_dir: Path to persistent storage
        local_dir: Path to local workspace
    """
    if not mount_gdrive():
        return None, Path.cwd()
    
    # Create project folder in Drive
    persistent_dir = Path(f"/content/drive/MyDrive/{project_name}")
    persistent_dir.mkdir(parents=True, exist_ok=True)
    
    local_dir = Path("/content/model-dqn-snn")
    
    print(f"📁 Persistent storage: {persistent_dir}")
    print(f"💻 Local workspace: {local_dir}")
    
    return persistent_dir, local_dir


def sync_checkpoint_to_drive(checkpoint_path, persistent_dir):
    """Copy checkpoint to Google Drive for persistence."""
    if not persistent_dir or not checkpoint_path.exists():
        return False
    
    try:
        drive_checkpoint = persistent_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, drive_checkpoint)
        print(f"☁️  Synced to Drive: {drive_checkpoint}")
        return True
    except Exception as e:
        print(f"⚠️  Drive sync failed: {e}")
        return False


def restore_checkpoint_from_drive(checkpoint_name, persistent_dir, local_dir):
    """Restore checkpoint from Google Drive after reconnect."""
    if not persistent_dir:
        return None
    
    drive_checkpoint = persistent_dir / checkpoint_name
    if not drive_checkpoint.exists():
        print(f"ℹ️  No checkpoint found in Drive: {checkpoint_name}")
        return None
    
    try:
        local_checkpoint = local_dir / "datasets" / "vision_unified" / checkpoint_name
        local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_checkpoint, local_checkpoint)
        print(f"✅ Restored checkpoint from Drive: {local_checkpoint}")
        return local_checkpoint
    except Exception as e:
        print(f"❌ Checkpoint restore failed: {e}")
        return None


def auto_sync_on_checkpoint(checkpoint_path):
    """
    Auto-sync checkpoint to Drive whenever saved.
    Call this after saving checkpoint in base_builder.py
    """
    # Try to sync to Drive if mounted
    try:
        from google.colab import drive
        persistent_dir = Path("/content/drive/MyDrive/model-dqn-snn")
        if persistent_dir.exists():
            sync_checkpoint_to_drive(checkpoint_path, persistent_dir)
    except:
        pass  # Not in Colab or Drive not mounted


# Usage in Colab notebook:
"""
# At the start of your Colab session:
from colab_persistence import setup_persistent_storage, restore_checkpoint_from_drive

# Setup
persistent_dir, local_dir = setup_persistent_storage()

# Try to restore previous checkpoint
restored = restore_checkpoint_from_drive(
    "encoding_checkpoint.pt", 
    persistent_dir, 
    local_dir
)

# Then run training as normal - it will auto-resume if checkpoint exists
# Checkpoints automatically sync to Drive every 2000 batches
"""
