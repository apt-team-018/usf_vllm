#!/bin/bash
# Wrapper script for install_nixl_from_source_ubuntu.py
# This script exists because the Dockerfile references install_nixl.sh
# but the actual implementation is in the Python script

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python installation script
python3 "${SCRIPT_DIR}/install_nixl_from_source_ubuntu.py" "$@"