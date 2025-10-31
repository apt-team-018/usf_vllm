# vLLM Docker Build Instructions

## Quick Start

Simply run the build script:

```bash
./build.sh
```

This will build the vLLM Docker image with all necessary configurations.

## Pushing to Docker Hub

After building, push the image to Docker Hub:

```bash
./push.sh
```

**Note**: You'll need to login to Docker Hub first if not already logged in:
```bash
docker login -u arpitsh018
```

---

## Build Configuration

The build script uses the following configuration:

- **CUDA Version**: 12.4.0
- **Python Version**: 3.11
- **Base Images**: `nvidia/cuda:12.4.0-devel-ubuntu22.04`
- **GPU Architecture**: 9.0 (optimized for H100 - Hopper GPUs)
- **Max Parallel Jobs**: 16
- **FlashInfer**: JIT compilation mode (AOT disabled)
- **Wheel Size Check**: Disabled
- **Target Stage**: `vllm-openai` (OpenAI-compatible API server)
- **Image Tag**: `vllm-usf:1.0.6`

---

## Manual Build Command

If you prefer to run the Docker command manually:

```bash
docker buildx build \
  --build-arg CUDA_VERSION=12.4.0 \
  --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 \
  --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg max_jobs=16 \
  --build-arg torch_cuda_arch_list='9.0' \
  --build-arg FLASHINFER_AOT_COMPILE=false \
  --build-arg RUN_WHEEL_CHECK=false \
  --tag vllm-usf:1.0.6 \
  --target vllm-openai \
  --load \
  --progress=plain \
  -f docker/Dockerfile \
  .
```

---

## Build Time Optimization

### Using Docker Cache (Recommended)

The build script does **NOT** include `--no-cache`, which means Docker will reuse cached layers from previous builds. This dramatically reduces build time:

- **First build**: ~6 hours (full compilation)
- **Subsequent builds**: ~5-10 minutes (only rebuilds changed layers)

### Force Clean Build

If you need to rebuild everything from scratch:

```bash
docker buildx build --no-cache \
  --build-arg CUDA_VERSION=12.4.0 \
  ... (rest of arguments)
```

⚠️ **Warning**: This will take ~6 hours!

---

## Post-Build Actions

### Running the Image

```bash
docker run --gpus all -p 8000:8000 vllm-usf:1.0.6
```

### Pushing to Docker Hub (Optional)

```bash
# Tag the image
docker tag vllm-usf:1.0.6 arpitsh018/vllm-usf:v1.0.6

# Push to Docker Hub
docker push arpitsh018/vllm-usf:v1.0.6
```

---

## Troubleshooting

### Common Issues

1. **"Failed to find the NIXL wheel"**
   - **Fixed**: Updated wheel search pattern in `tools/install_nixl_from_source_ubuntu.py`
   - The script now searches for both `nixl-*.whl` and `nixl_cu*-*.whl` patterns

2. **"apache-tvm-ffi dependency error"**
   - **Fixed**: Added `--prerelease=allow` flag to all `uv pip install` commands in Dockerfile
   - This allows installation of pre-release packages required by FlashInfer

3. **"install_nixl.sh not found"**
   - **Fixed**: Created wrapper script at `tools/install_nixl.sh`
   - Also added COPY instruction in Dockerfile for the Python script

### Build Logs

For detailed build output, the `--progress=plain` flag is already included in the build command.

---

## GPU Architecture Support

The current build is optimized for **H100 GPUs** (compute capability 9.0).

### For Other GPU Architectures

Modify the `torch_cuda_arch_list` build argument:

```bash
# For multiple architectures (slower build, broader compatibility)
--build-arg torch_cuda_arch_list='7.0 7.5 8.0 8.9 9.0 10.0 12.0'

# For A100 GPUs (compute capability 8.0)
--build-arg torch_cuda_arch_list='8.0'

# For V100 GPUs (compute capability 7.0)
--build-arg torch_cuda_arch_list='7.0'
```

---

## Files Modified

This build includes the following fixes:

1. **`docker/Dockerfile`**
   - Added `--prerelease=allow` to uv pip install commands (lines 145, 175, 360, 406, 413, 433)
   - Added COPY instruction for `install_nixl_from_source_ubuntu.py` (line 445)

2. **`tools/install_nixl.sh`** (created)
   - Wrapper script for NIXL installation

3. **`tools/install_nixl_from_source_ubuntu.py`**
   - Updated wheel search patterns to match actual filenames
   - Added temp directory creation

4. **`build.sh`** (created)
   - Automated build script with all configurations

---

## Support

For issues or questions, refer to the [vLLM GitHub repository](https://github.com/vllm-project/vllm).