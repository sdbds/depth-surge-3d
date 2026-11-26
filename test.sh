#!/bin/bash
set -u

LIME='\033[38;2;57;255;20m'
CYAN='\033[38;2;0;217;255m'
RED='\033[38;2;255;0;0m'
NC='\033[0m'

echo -e "${CYAN}Testing Depth Surge 3D setup...${NC}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Follow docs/INSTALLATION.md first."
    exit 1
fi

FAILED=0

echo -e "\n${CYAN}Testing Python imports and native Quality extension...${NC}"
python -c "
import torch
import cv2
import numpy as np
import flask
import flask_socketio
import depth_surge_3d
import _quality_kd_native

print('OK PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('OK CUDA device:', torch.cuda.get_device_name(0))
print('OK OpenCV:', cv2.__version__, 'NumPy:', np.__version__)
print('OK depth_surge_3d:', depth_surge_3d.__version__)
print('OK _quality_kd_native:', _quality_kd_native.__file__)
" || FAILED=1

echo ""
for tool in ffmpeg ffprobe; do
    if command -v "$tool" &> /dev/null; then
        echo -e "${LIME}OK${NC} $tool is available"
    else
        echo -e "${RED}FAIL${NC} $tool not found on PATH"
        FAILED=1
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo -e "${LIME}Setup test passed!${NC}"
    echo "Next steps:"
    echo "  Web UI:  ./run_ui.sh"
    echo "  CLI:     depth-surge-3d input_video.mp4"
else
    echo -e "${RED}Setup test failed. Check errors above.${NC}"
fi
exit "$FAILED"
