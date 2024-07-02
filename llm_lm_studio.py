import llm
from openai import OpenAI
import httpx

@llm.hookimpl
def register_models(register):
    register(LMStudio())

class LMStudio(llm.Model):
    model_id = "lm-studio"
    can_stream = True

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

    def get_first_model(self):
        resp = httpx.get("http://localhost:1234/v1/models")
        models = resp.json()
        return models['data'][0]['id']

    def execute(self, prompt, stream, response, conversation):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        response = client.chat.completions.create(
            model=self.get_first_model(),
            messages=self.build_messages(prompt, conversation),
            temperature=0.8,
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content
