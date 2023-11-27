import requests


def flag_prompt(prompt: str, lakera_guard_api_key):
    response = requests.post(
        "https://api.lakera.ai/v1/prompt_injection",
        json={"input": prompt},
        headers={"Authorization": f"Bearer {lakera_guard_api_key}"},
    ).json()

    flagged = response["results"][0]["flagged"]
    return flagged, response
