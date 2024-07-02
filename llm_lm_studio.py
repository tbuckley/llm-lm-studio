import llm

@llm.hookimpl
def register_models(register):
    register(Markov())

class Markov(llm.Model):
    model_id = "lm-studio"

    def execute(self, prompt, stream, response, conversation):
        return ["hello world"]