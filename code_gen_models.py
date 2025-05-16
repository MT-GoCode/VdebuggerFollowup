from abc import ABC, abstractmethod

class CodeGenModel(ABC):
    # no more GPU number... just set CUDA_VISIBLE_DEVICES, and these models will try to use all of them.

    def __init__(self, code_generation_model_config):
        self.config = code_generation_model_config

    def generate(self, base_prompt, batch):
        print("Prompting code gen model...")

        batch_filled_prompts = []

        for i, id in enumerate(batch['id']):
            prompt = base_prompt

            for key in batch.keys():
                if key == 'id': continue
                placeholder = f"${key}$"
                prompt = prompt.replace(placeholder, batch[key][i])
            batch_filled_prompts.append(prompt)
        
        result = self._generate(batch_filled_prompts)
        return result, batch_filled_prompts
    
    @abstractmethod
    def _generate(batch_filled_prompts):
        pass

import openai
from functools import partial
import os

def call_openai(config, prompt):

    # Config is assumed to be global or passed in via partial
    params = {
        "model": config.specific_model,
        "messages": [
            {"role": "system", "content": "You are a coding assistant will align your responses exactly to textual requirements such as function headers and return statements."},
            {"role": "user", "content": prompt}
        ],
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    if config.temperature is not None:
        params["temperature"] = config.temperature
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop is not None:
        params["stop"] = list(config.stop)

    # client initialization must be done in each thread, because there's a lock.
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_KEY")
    )

    response = client.chat.completions.create(**params)
    return response.choices[0].message.content

class OpenAIAPICategory(CodeGenModel):
    name = 'openai'

    def __init__(self, code_generation_model_config):
        super().__init__(code_generation_model_config=code_generation_model_config)

    def _generate(self, batch_filled_prompts):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            resp = list(executor.map(partial(call_openai, self.config), batch_filled_prompts))
                
        print("raw code generation output")
        print(resp)

        # cleanup - ensure function bodies are returned.
        """
        Expected response is something like

        ```python\ndef execute_command(video, possible_answers, query) -> [str, dict]:\n    video_segment = VideoSegment(video)\n    # Find the frame where the two men are playing the instrument\n    frame_of_interest = None\n    for i, frame in enumerate(video_segment.frame_iterator()):\n        if frame.exists("man") and frame.simple_query("Are the men playing an instrument?") == "yes":\n            frame_of_interest = frame\n            break\n    if frame_of_interest is None:\n        # Select frame at the middle of the video, as no temporal info is available\n        frame_of_interest = video_segment.frame_from_index(video_segment.num_frames // 2)\n    # Caption the frame\n    caption = frame_of_interest.simple_query("What is in the frame?")\n    # Determine how the men are playing the instrument\n    play_method = frame_of_interest.simple_query("How are the men playing the instrument?")\n    # Create the info dictionary\n    info = {\n        "Caption of frame with men playing instrument": caption,\n        "Method of playing detected": play_method\n    }\n    # Answer the query\n    answer = video_segment.select_answer(info, query, possible_answers)\n    return answer

        so we need to strip first two lines. \n```follows but is excluded!
        """

        for i in range(len(resp)):
            resp[i] = "\n".join(resp[i].splitlines()[2:])

        return resp
   
import torch

class VLLMCategory(CodeGenModel):
    name = 'vllm'

    def __init__(self, code_generation_model_config):
        super().__init__(code_generation_model_config=code_generation_model_config)

        from vllm import LLM

        model_id = self.config.specific_model

        if not os.path.exists(model_id) and os.path.isdir(model_id):
            assert model_id in [
                'codellama/CodeLlama-7b-hf', 'codellama/CodeLlama-13b-hf', 'codellama/CodeLlama-34b-hf',
                'codellama/CodeLlama-7b-Python-hf', 'codellama/CodeLlama-13b-Python-hf',
                'codellama/CodeLlama-34b-Python-hf', 'codellama/CodeLlama-7b-Instruct-hf',
                'codellama/CodeLlama-13b-Instruct-hf', 'codellama/CodeLlama-34b-Instruct-hf',
                'codellama/CodeLlama-70b-Python-hf', 'deepseek-ai/deepseek-coder-33b-base',
                'deepseek-ai/DeepSeek-Coder-V2-Lite-Base',
            ]
        # Note: 70b-Instruct-hf has special formatting, will handle in the future
        self.is_instruct = 'Instruct' in model_id
        self.llm = LLM(model=model_id, dtype='bfloat16', tensor_parallel_size=torch.cuda.device_count(),
                       max_model_len=10000, trust_remote_code=True)
        self.sampling_params = self.get_sampling_params()

    def get_sampling_params(self):
        from vllm import SamplingParams

        num_return_sequences = 1
        return SamplingParams(
            n=num_return_sequences,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            stop=["\n\n"],
        )

    def general_complete(self, prompt):
        outputs = self.llm.generate(prompt, self.sampling_params, use_tqdm=True)
        generated_text = [[o.text for o in output.outputs] for output in outputs]
        generated_text = [[text.split('\n\n')[0] for text in texts] for texts in generated_text]

        assert all(len(texts) == 1 for texts in generated_text)
        generated_text = [texts[0] for texts in generated_text]
        return generated_text
    
    def _generate(self, batch_filled_prompts):
        resp = self.general_complete(batch_filled_prompts)
        return resp
    

        # if self.is_instruct:  # ridiculous.
        #     generated_text_ = []
        #     for texts in generated_text:
        #         generated_text_.append([])
        #         for text in texts:
        #             if self.FUNCTION_HEAD in text:
        #                 text = self.FUNCTION_HEAD + text.split(self.FUNCTION_HEAD)[1]
        #             else:
        #                 text = self.FUNCTION_HEAD
        #             if "```" in text:
        #                 text = text.split("```")[0]
        #             generated_text_[-1].append(text)
        #     generated_text = generated_text_
        # if self.is_instruct:
        #     B_INST, E_INST = "[INST]", "[/INST]"
        #     B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        #     prompt = [B_SYS + self.SYSTEM + E_SYS + p for p in prompt]
        #     prompt = [f"{B_INST} {p.strip()} {E_INST}" for p in prompt]