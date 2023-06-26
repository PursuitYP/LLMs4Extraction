import os
import openai
from dotenv import load_dotenv
import requests

load_dotenv()

# os.environ["http_proxy"] = "http://10.10.1.3:10000"
# os.environ["https_proxy"] = "http://10.10.1.3:10000"

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

# response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)
# print(response)

def get_item_package(role, content):
    return {"role": role, "content": content}


def get_response(messages):
    resp = requests.post(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai.api_key}"},
        json={
            "model": "gpt-3.5-turbo",
            "messages": messages,
        },
        proxies={
            "http": "http://10.10.1.3:10000",
            "https": "http://10.10.1.3:10000",
        }
    )
    return resp.json()


def main_session():
    messages = [
        get_item_package("system", "You are a helpful assistant."),
    ]
    while True:
        user_input = input("User: ")
        messages.append(get_item_package("user", user_input))
        # res = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        res = get_response(messages)
        cur_content = res['choices'][0]['message']['content']
        print(cur_content, "\n")
        messages.append(get_item_package("assistant", cur_content))

main_session()
# messages = ["please explain TCP in 40 words",
#             "please explain UDP in 40 words",
#             "please compare TCP and UDP in 80 words"]
