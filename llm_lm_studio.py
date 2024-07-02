import llm
from openai import OpenAI

@llm.hookimpl
def register_models(register):
    register(LMStudio())

class LMStudio(llm.Model):
    model_id = "lm-studio"
    can_stream = True

    def execute(self, prompt, stream, response, conversation):
        messages = []
        if prompt.system is not None:
            messages.append({"role": "system", "content": prompt.system})
        if conversation is not None:
            for response in conversation.responses:
                messages.append({"role": "user", "content": response.prompt.prompt})
                messages.append({"role": "assistant", "content": response.text()})
        messages.append({"role": "user", "content": prompt.prompt})

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        response = client.chat.completions.create(
            model="bartowski/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf",
            messages=messages,
            temperature=0.7,
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content
