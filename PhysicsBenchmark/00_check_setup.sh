#!/bin/bash
# ============================================================
#  WorldGen Setup Checker
#  Run this FIRST to verify your environment is ready
# ============================================================

echo "======================================================"
echo "  WorldGen Environment Check"
echo "======================================================"

PASS=0; FAIL=0

check() {
    local name="$1"; local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo "  [OK]  $name"
        ((PASS++))
    else
        echo "  [MISSING] $name"
        ((FAIL++))
    fi
}

echo ""
echo "--- System Tools ---"
check "Python 3"        "python3 --version"
check "pip3"            "pip3 --version"
check "Blender"         "blender --version"

echo ""
echo "--- Python Packages ---"
check "numpy"           "python3 -c 'import numpy'"
check "scipy"           "python3 -c 'import scipy'"
check "matplotlib"      "python3 -c 'import matplotlib'"
check "pybullet"        "python3 -c 'import pybullet'"
check "trimesh"         "python3 -c 'import trimesh'"
check "tqdm"            "python3 -c 'import tqdm'"

echo ""
echo "======================================================"
echo "  Result: $PASS OK  |  $FAIL MISSING"
echo "======================================================"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "  Run this to fix missing packages:"
    echo "  pip3 install pybullet trimesh numpy scipy matplotlib tqdm"
    echo ""
    echo "  Blender installation (Ubuntu/Debian):"
    echo "  sudo apt update && sudo apt install -y blender"
    echo "  OR download from: https://www.blender.org/download/"
fi
