#!/bin/bash
# Wrapper script for install_nixl_from_source_ubuntu.py
# This script exists because the Dockerfile references install_nixl.sh
# but the actual implementation is in the Python script

set -e

# In the Docker container, the Python script will be in the same directory as this script
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python installation script from the same directory
python3 "${SCRIPT_DIR}/install_nixl_from_source_ubuntu.py" "$@"