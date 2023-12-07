# daltunay

Access live deployed app here: **https://daltunay.streamlit.app/**

## Prerequisites

**Poetry**: If [Poetry](https://python-poetry.org/) is not installed, you can do so using pip:


```bash
pip install poetry
```

**Docker**: If [Docker](https://www.docker.com/) is not installed, you can do so following [this link](https://docs.docker.com/get-docker/)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/daltunay/my-app.git
cd daltunay
```

2. Set up the project dependencies using Poetry:

```bash
poetry install
```

This command will create a virtual environment and install the necessary dependencies.

## Setting up API Keys

The application uses several APIs to function properly. 
You can specifiy the API keys in `.streamlit/secrets.toml`: 

```toml
[openai_api]
key="<OPENAI_API_KEY>"

[together_api]
key="<TOGETHER_API_KEY>"

[lakera_api]
key="<LAKERA_GUARD_API_KEY>"

[google_api]
key = "<GOOGLE_API_KEY>"
cse_id = "<GOOGLE_CSE_ID>"
```


## Running the Application
The _daltunay_ application can be run using either Poetry or Docker.

### Using Poetry

To run the application using Poetry:

```bash
poetry run streamlit run app.py
```

### Using Docker

1. Build the Docker image:

```bash
docker build -t daltunay .
```

2. Run the application as a Docker container:

```bash
docker run -p 8501:8501 daltunay
```

Alternatively, you can just run the following:

```bash
chmod +x ./bin/run.sh
./bin/run.sh
```

Once the application is running, it will be accessible at http://localhost:8501 in your web browser.
