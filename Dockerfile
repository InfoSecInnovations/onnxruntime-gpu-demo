FROM astral/uv:0.11.1-python3.13-trixie-slim

# lines below pasted from uv documentation
RUN uv venv /opt/venv
# Use the virtual environment automatically
ENV VIRTUAL_ENV=/opt/venv
# Place entry points in the environment at the front of the path
ENV PATH="/opt/venv/bin:$PATH"
# end uv documentation pasting

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# enable nvidia packages
RUN sed -i -e's/ main/ main contrib non-free/g' /etc/apt/sources.list.d/debian.sources
ADD https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

# this appears to be sufficient to get cuda working with fastembed
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
/bin/sh -c set -eux; \ 
apt-get update; \
apt-get install -y --no-install-recommends build-essential cuda-toolkit

# recommended for uv in a container
ENV UV_LINK_MODE=copy
# try to help with the large nvidia downloads timing out
ENV UV_HTTP_TIMEOUT=500

# this version of PyTorch is compatible with older GPUs
RUN --mount=type=cache,target=/root/.cache/uv uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
RUN --mount=type=cache,target=/root/.cache/uv uv pip install sentence-transformers~=5.2.3 langchain-text-splitters~=1.1.1 requests fastembed-gpu tika

COPY . .

CMD ["uv", "run", "tika_ingest.py"]