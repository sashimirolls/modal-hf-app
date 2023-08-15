import requests
import json
url_aws = 'https://d5b0183h6j.execute-api.eu-central-1.amazonaws.com/beta'

headers = {
    "x-api-key" :"gkBrri7sq88IC0ArmoR6R28aTjM8qfji9BGwHcHs",
    "Content-Type": "application/json"}
string ="""### human: Act as a girlfriend.

### response: Sure! sound good.

### human: hi

### response:"""
json = {"prompt": string,
        "temperature":0.4,
        "max_tokens":800}
response = requests.post(url_aws, headers=headers,json=json)

response_json = str(response.json())


if response.status_code == 200:
    print("Response:", response_json)
else:
    print("Request failed with status code:", response.json())