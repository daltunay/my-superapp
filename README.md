
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-science-superapp.streamlit.app)


## Prerequisites

**Poetry**: If [Poetry](https://python-poetry.org/) is not installed, you can do so using pip:


```bash
pip install poetry
```

**Docker**: If [Docker](https://www.docker.com/) is not installed, you can do so following [this link](https://docs.docker.com/get-docker/)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/daltunay/my-superapp.git
cd my-superapp
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
[twilio]
TWILIO_ACCOUNT_SID = "<...>"
TWILIO_AUTH_TOKEN = "<...>"

[openai]
OPENAI_API_KEY = "<...>"

[together]
TOGETHER_API_KEY = "<...>"

[lakera_guard]
LAKERA_GUARD_API_KEY = "<...>"

[google]
GOOGLE_API_KEY = "<...>"
GOOGLE_CSE_ID = "<...>"
```


## Running the Application
The _my-superapp_ application can be run using either Poetry or Docker.

### Using Poetry

To run the application using Poetry:

```bash
poetry run streamlit run app.py
```

### Using Docker

1. Build the Docker image:

```bash
docker build -t my-superapp .
```

2. Run the application as a Docker container:

```bash
docker run -p 8501:8501 my-superapp
```

Alternatively, you can just run the following:

```bash
chmod +x ./bin/run.sh
./bin/run.sh
```

Once the application is running, it will be accessible at http://localhost:8501 in your web browser.
