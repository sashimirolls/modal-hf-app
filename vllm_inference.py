# # Fast inference with vLLM (Llama 2 13B)

# To run
# [any of the other supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html),
# simply replace the model name in the download step. You may also need to enable `trust_remote_code` for MPT models (see comment below)..
#
# ## Setup
#
# First we import the components we need from `modal`.

import os
from typing import Dict
from modal import Image, Secret, Stub, method, web_endpoint


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
#
# Next, [create a HuggingFace access token](https://huggingface.co/settings/tokens).
# To access the token in a Modal function, we can create a secret on the [secrets page](https://modal.com/secrets).
# Now the token will be available via the environment variable named `HUGGINGFACE_TOKEN`. Functions that inject this secret will have access to the environment variable.
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# Tip: avoid using global variables in this function. Changes to code outside this function will not be detected and the download step will not re-run.
def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "NousResearch/Redmond-Puffin-13B",
        local_dir="/model",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


MODEL_DIR = "/model"

# ### Image definition
# We’ll start from a Dockerhub image recommended by `vLLM`, upgrade the older
# version of `torch` to a new one specifically built for CUDA 11.8. Next, we install `vLLM` from source to get the latest updates.
# Finally, we’ll use run_function to run the function defined above to ensure the weights of the model
# are saved within the container image.
#
image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pin vLLM to 07/19/2023
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@bda41c70ddb124134935a90a0d51304d2ac035e8"
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secret=Secret.from_name("huggingface"),
        timeout=60 * 20,
    )
)

stub = Stub("llama-vllm-inference", image=image)


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `__enter__` method.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean.

@stub.cls(gpu="A100", secret=Secret.from_name("huggingface"),container_idle_timeout=180)
class Model:
    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR)
        self.template = """{}"""

    @method()
    def generate(self, user_questions, temperature:float, max_tokens:int):
        from vllm import SamplingParams

        prompts = [self.template.format(user_questions)]
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1,
            max_tokens=max_tokens,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        num_tokens = 0
        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(output.prompt, output.outputs[0].text, "\n\n", sep="")
        print(f"Generated {num_tokens} tokens")
        return output.outputs[0].text
    

# ## Function to test locally
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@stub.local_entrypoint()
def main():
    model = Model()
    questions = "### human: Tell me your name\n\n### response:"
    model.generate.call(questions)

#Web function for API
@stub.function(container_idle_timeout=300)
@web_endpoint(method="POST")
def predict(input:Dict):
    response=""
    try:
        prompt= input.get("prompt")
        if (prompt and prompt != ""):
            temp=input.get("temperature",0.75)
            tokens = input.get("max_tokens",800)
            model = Model()
            response = model.generate.call(prompt,temp,tokens)
        else:
            response = "Prompt template cannot be empty"
    except Exception as e:
        response= "Invalid json format"


    return response

