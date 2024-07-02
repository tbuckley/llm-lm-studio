import llm
from openai import OpenAI
import httpx
from typing import Optional
from pydantic import field_validator, Field

@llm.hookimpl
def register_models(register):
    register(LMStudio())

class LMStudio(llm.Model):
    model_id = "lm-studio"
    can_stream = True

    class Options(llm.Options):
        port: Optional[int] = Field(
            description="Port for LM Studio server",
            default=None,
        )

        @field_validator("port")
        def validate_port(cls, port):
            if port is None:
                return None
            if port < 1 or port > 65535:
                raise ValueError("invalid port number")
            return port

    def build_messages(self, prompt, conversation):
        messages = []
        if prompt.system is not None:
            messages.append({"role": "system", "content": prompt.system})
        if conversation is not None:
            for response in conversation.responses:
                messages.append({"role": "user", "content": response.prompt.prompt})
                messages.append({"role": "assistant", "content": response.text()})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def get_first_model(self, port):
        resp = httpx.get(f"http://localhost:{port}/v1/models")
        models = resp.json()
        return models['data'][0]['id']

    def execute(self, prompt, stream, response, conversation):
        port = prompt.options.port or 1234
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="lm-studio")
        response = client.chat.completions.create(
            model=self.get_first_model(port),
            messages=self.build_messages(prompt, conversation),
            temperature=0.8,
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content
