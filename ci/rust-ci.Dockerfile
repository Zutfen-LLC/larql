# Base CI image for larql Rust workflows.
#
# Extends the official rust:1-bookworm image with sccache pre-installed,
# eliminating a per-run curl download. The cargo registry / dependency
# sources are NOT baked in here — they persist via a named Docker volume
# mounted at /usr/local/cargo/registry in each workflow.
#
# Rebuild: triggered automatically when Rust minor version or sccache
# version changes (see .github/workflows/ci-image.yml).

FROM rust:1-bookworm

ARG SCCACHE_VERSION=0.16.0

# sccache with S3 backend (Debian bookworm package lacks S3 support).
RUN curl -sL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz -C /tmp \
 && mv "/tmp/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache" /usr/local/bin/sccache \
 && chmod +x /usr/local/bin/sccache

# Create the cargo registry directory so the volume mount target exists
# and is owned by root (matching the container user) on first run.
RUN mkdir -p /usr/local/cargo/registry

LABEL org.opencontainers.image.title="larql-ci-rust"
LABEL org.opencontainers.image.description="Rust CI base with sccache for larql"
LABEL org.opencontainers.image.source="https://github.com/Zutfen-LLC/larql"
