import requests
import json
url_aws = ''

headers = {
    "x-api-key" :"",
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