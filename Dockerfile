From python:3.9.1-slim-buster

# WORKDIR ./
COPY . .

# Install dependencies using pip inside the virtual environment
RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config python3-dev
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main app file to the container
COPY main.py .

EXPOSE 5050
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]
