FROM python:3.13-slim
WORKDIR /app

# Upgrade pip and setuptools, and install hatch
RUN python -m pip install --upgrade pip setuptools && \
    pip install --upgrade wheel hatch

# Copy the FastAPI app code into the container
COPY . .

# Install dependencies using hatch
RUN hatch env create production

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["hatch", "run", "serve"]
