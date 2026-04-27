import logging
import re
from abc import abstractmethod, ABC
from collections.abc import Iterable
from os import getenv
from pathlib import Path
from typing import Generator

from jinja2 import Environment, FileSystemLoader
from transformers import pipeline

from .common import InputItem, OutputItem

logger = logging.getLogger(__name__)
logging.basicConfig(level=getenv("LOGLEVEL", "INFO").upper())


class SyllogismSolver(ABC):
    """A syllogism solver."""

    def __init__(
        self, prompt_path: str = "solver.j2", config: str | None = None, *args, **kwargs
    ) -> None:
        self.pipe = pipeline("text-generation", *args, **kwargs)

        env = Environment(loader=FileSystemLoader(Path(__file__) / "prompts"))
        self.prompt = env.get_template(prompt_path)

    def solve(self, data: Iterable[InputItem]) -> Generator[OutputItem, None, None]:
        for item in data:
            messages = [
                {
                    "role": "system",
                    "content": self.prompt.render(),
                },
                {
                    "role": "user",
                    "content": item["syllogism"],
                },
            ]
            response = self.pipe(messages)
            logger.info(response)
            answer = response[0]["generated_text"][-1]["content"]

            yield {"id": item["id"], "validity": self._extract_validity(answer)}

    @abstractmethod
    def _extract_validity(self, answer: str) -> bool:
        ...


class PeftSyllogismSolver(SyllogismSolver):
    """A syllogism solver which uses PeftModel."""

    def __init__(self, model_name: str, adapter_name: str) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .training.lora import bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter_name)
        tokenizer = AutoTokenizer.from_pretrained(adapter_name)

        super().__init__(model=model, tokenizer=tokenizer)

    def _extract_validity(self, answer: str) -> bool:
        return "TRUE" in answer


class PeftThinkingSyllogismSolver(SyllogismSolver):
    """A syllogism solver which uses PeftModel."""

    def __init__(self, model_name: str, adapter_name: str) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .training.grpo_lora import bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter_name)
        tokenizer = AutoTokenizer.from_pretrained(adapter_name)

        super().__init__(prompt_path="solver_thinking.j2", model=model, tokenizer=tokenizer)

    def _extract_validity(self, answer: str) -> bool:
        answer = answer.strip()
        last_sentence = answer.split(".")[-1]
        return re.search(r"\bvalid\b", last_sentence) is not None
