#!/bin/bash
# Optimized Colab Setup Script
# Handles: Drive mounting, checkpoint restoration, efficient installation

set -e  # Exit on error

echo "========================================================================"
echo "  Colab Setup: TRM Training Pipeline"
echo "========================================================================"

# 1. Mount Google Drive for persistence
echo "📁 Mounting Google Drive..."
python3 -c "
from google.colab import drive
import os
drive.mount('/content/drive', force_remount=False)
os.makedirs('/content/drive/MyDrive/model-dqn-snn', exist_ok=True)
print('✅ Drive mounted and folder ready')
"

# 2. Clone/update repository
echo ""
echo "📦 Setting up repository..."
if [ -d "model-dqn-snn" ]; then
    echo "   Repository exists, pulling latest changes..."
    cd model-dqn-snn
    git pull
    cd ..
else
    echo "   Cloning repository..."
    git clone https://github.com/saoxdxd2/model-dqn-snn.git
fi

cd model-dqn-snn

# 3. Install dependencies efficiently
echo ""
echo "📦 Installing dependencies..."
pip install -q --no-cache-dir -r requirements.txt

# 4. Restore checkpoint if exists
echo ""
echo "🔄 Checking for saved checkpoint..."
python3 -c "
import shutil
from pathlib import Path

drive_checkpoint = Path('/content/drive/MyDrive/model-dqn-snn/encoding_checkpoint.pt')
local_checkpoint = Path('datasets/vision_unified/encoding_checkpoint.pt')

if drive_checkpoint.exists():
    local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drive_checkpoint, local_checkpoint)
    print('✅ Restored checkpoint from Drive')
    print(f'   Location: {local_checkpoint}')
else:
    print('ℹ️  No previous checkpoint found - starting fresh')
"

# 5. Start training
echo ""
echo "🚀 Starting training pipeline..."
echo "   Auto-saves every 2000 batches (~1 hour)"
echo "   Checkpoints sync to Google Drive automatically"
echo "========================================================================"
echo ""

python train.py
