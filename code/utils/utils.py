import time
import json
from collections import defaultdict
import argparse
from openai import OpenAI


client_local = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)


def ask_gpt(messages, use_temp=1, max_token=128, do_sample=True, model_name="meta-llama/Llama-3.1-70B-Instruct"):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>".format(
                message["content"])
        elif message["role"] == "user":
            prompt += "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>".format(message["content"])
        elif message["role"] == "assistant":
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>".format(message["content"])
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    if do_sample:
        while True:
            try:
                r = client_local.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_token,
                    temperature=use_temp
                )
                return r.choices[0].text

            except Exception as e:
                if "less than 1024 tokens" in str(e):
                    return ""
                time.sleep(1)
                continue
    else:
        while True:
            try:
                r = client_local.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_token,
                    temperature=use_temp
                )
                return r.choices[0].text
            except Exception as e:
                if "less than 1024 tokens" in str(e):
                    return ""
                continue