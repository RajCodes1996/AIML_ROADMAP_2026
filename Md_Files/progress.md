4. Part C — Docker fundamentals
Now comes an important concept in AI/ML engineering.
What is Docker?
Suppose your API works perfectly on your Windows laptop.

But when you move it to another computer, it might fail because:
- Python versions are different.
- Required libraries are missing.
- Dependencies conflict.
- Files are located in different places.
Docker helps solve this problem by packaging your application and its dependencies into a portable container image.

Think of it this way:
- Dockerfile: Instructions for packaging your application.
- Docker image: The packaged blueprint.
- Docker container: A running instance of that image.








mlrepa.github.io


Step 1: Install Docker Desktop
If Docker Desktop is not installed, install it for Windows:
Download Docker Desktop for WindowsAfter installation, open Docker Desktop and wait until its engine is running.

Check Docker in your VS Code terminal:
docker --version


Then:
docker info


The second command should display information about the running Docker engine.
Step 2: Create a Dockerfile
Create a file named exactly:
Dockerfile


Place it in the root of Fraud_Detection_Project, alongside requirements.txt.

Use this:

# Use Python 3.11 as the base environment.
# Change this version if your local Python version differs.
FROM python:3.11-slim

# Set application working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Copy saved ML model
COPY models/ ./models/

# API listens on port 8000
EXPOSE 8000

# Start FastAPI using Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]



Important compatibility note: The Python and scikit-learn versions used in Docker should match the environment used to save your Joblib model as closely as possible. Otherwise, model-loading compatibility problems can occur. Check your local versions using:
python --version
python -c "import sklearn; print(sklearn.__version__)"


If your Python version is not 3.11, change the Dockerfile's base image accordingly. Also pin your installed scikit-learn version in requirements.txt to the version you used to train the model.
Step 3: Create .dockerignore
Create another file in the project root:
.dockerignore


Add:
.venv/
venv/
__pycache__/
.pytest_cache/
.git/
.vscode/
*.pyc


This prevents unnecessary development files from being copied into the Docker build.
Step 4: Build your Docker image
Make sure your terminal is in the project root.

Run:
docker build -t fraud-detection-api:v1 .


What happens?
1. Docker reads your Dockerfile.
2. Downloads the Python base image if needed.
3. Installs your dependencies.
4. Copies your FastAPI code.
5. Copies your saved ML model.
6. Creates a Docker image.
Check the image:
docker images


You should find:
fraud-detection-api


Step 5: Run your Docker container
Execute:
docker run -d --name fraud-api -p 8000:8000 fraud-detection-api:v1


Understand the command:
Part	Meaning
-d	Run in the background
--name fraud-api	Give the container a name
-p 8000:8000	Connect your computer's port 8000 to the container's port 8000
fraud-detection-api:v1	Image to run

Now open:
Open your Dockerized API documentationTry the /health and /predict endpoints again.

If the health endpoint returns:
{
  "status": "healthy",
  "model_loaded": true
}


your API is running inside Docker!

Note: Stop your local Uvicorn server before running the container, otherwise both applications may try to use port 8000.

Useful Docker commands:
# See running containers
docker ps

# See application logs
docker logs fraud-api

# Stop container
docker stop fraud-api

# Start it again
docker start fraud-api

# Remove stopped container
docker rm -f fraud-api


5. Final project structure
After Day 24, your project should look like this:
Fraud_Detection_Project/
│
├── models/
│   └── fraud_detection_v1.joblib
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── app.py
│
├── tests/
│   └── test_app.py
│
├── reports/
│   └── evaluation.txt
│
├── Dockerfile
├── .dockerignore
└── requirements.txt