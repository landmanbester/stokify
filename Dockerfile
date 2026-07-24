FROM python:3.11-slim

ARG HIP_CARGO_REF=apis

WORKDIR /app

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /usr/local/bin/uv

# stokify tracks hip-cargo's unreleased apis-branch APIs via a side-by-side
# path source ([tool.uv.sources] -> ../hip-cargo). Recreate that layout inside
# the image from a GitHub tarball (no git needed). /app/../hip-cargo == /hip-cargo.
ADD https://github.com/landmanbester/hip-cargo/archive/refs/heads/${HIP_CARGO_REF}.tar.gz /tmp/hip-cargo.tar.gz
RUN mkdir -p /hip-cargo \
    && tar -xzf /tmp/hip-cargo.tar.gz -C /hip-cargo --strip-components=1 \
    && rm /tmp/hip-cargo.tar.gz

# Copy package files (LICENSE is required by pyproject's license-files glob)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install package with full dependencies
RUN uv pip install --system --no-cache ".[full]"

# Make CLI available
CMD ["stokify", "--help"]
