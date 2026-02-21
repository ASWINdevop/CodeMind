# 1. Base Image: High-efficiency Linux environment
FROM python:3.11-slim

# 2. System Dependencies: Install Git for repo cloning
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

# 3. Establish Working Directory
WORKDIR /app

# 4. Install Python Dependencies
COPY requirements.txt .

# --- THE 10GB FIX: Force CPU-only PyTorch BEFORE processing requirements ---
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Source Code
COPY src/ ./src/

# 6. Expose Streamlit Port
EXPOSE 8501

# --- FIX: FORCE PYTHON TO RECOGNIZE THE ROOT DIRECTORY ---
ENV PYTHONPATH="/app"

# 7. Execution Command
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]