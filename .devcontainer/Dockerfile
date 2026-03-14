# Python 3.14 Bookworm
FROM python:3.14-bookworm

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Setup profile.sh
COPY profile.sh /root/profile.sh
RUN echo ". profile.sh" >> "/root/.bashrc"

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY uv.lock* ./
COPY README.md ./

# Sync dependencies
RUN uv sync --frozen --no-install-project || uv sync --no-install-project

# Copy the rest
COPY . .

# Sync the project
RUN uv sync --frozen || uv sync

# Set entrypoint if needed, or default
CMD ["/bin/bash"]
