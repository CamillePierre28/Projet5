# Use a lightweight Python 3.12 image
FROM python:3.12-slim

# Set working directory
WORKDIR /code

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create non-root user (best practice for Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory for user
WORKDIR $HOME/app

# Copy project files
COPY --chown=user . $HOME/app

# Launch FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]