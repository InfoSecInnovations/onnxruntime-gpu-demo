FROM astral/uv:0.11.1-python3.13-trixie-slim

# lines below pasted from uv documentation
RUN uv venv /opt/venv
# Use the virtual environment automatically
ENV VIRTUAL_ENV=/opt/venv
# Place entry points in the environment at the front of the path
ENV PATH="/opt/venv/bin:$PATH"

# recommended for uv in a container
ENV UV_LINK_MODE=copy
# try to help with the large nvidia downloads timing out
ENV UV_HTTP_TIMEOUT=500

# this version of PyTorch is compatible with older GPUs
RUN uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
RUN uv pip install onnxruntime-gpu

COPY ./check_cuda.py ./check_cuda.py

CMD ["uv", "run", "check_cuda.py"]