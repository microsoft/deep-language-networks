import requests


def download_json_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    json_data = response.json()
    return json_data
