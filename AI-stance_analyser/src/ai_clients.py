import os
import time
import random

USE_NEW_CLIENT = os.getenv("USE_NEW_GENAI", "1") == "1"

RETRY_STATUS = {"UNAVAILABLE", "RESOURCE_EXHAUSTED"}
MAX_RETRIES = int(os.getenv("GENAI_MAX_RETRIES", "5"))
BASE_DELAY = float(os.getenv("GENAI_BASE_DELAY", "1.0"))
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")

class GeminiClient:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")

        if USE_NEW_CLIENT:
            from google import genai
            self.genai = genai
            self.client = genai.Client(api_key=self.api_key)
            self.mode = "new"
        else:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.model = genai.GenerativeModel(self.model_name)
            self.mode = "legacy"

    def _call_once(self, model: str, contents, system_instruction: str, temperature: float, max_output_tokens: int):
        if self.mode == "new":
            return self.client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",
                },
            ).text
        else:
            mdl = self.model if model == self.model_name else self.genai.GenerativeModel(model)
            return mdl.generate_content(
                contents if isinstance(contents, str) else contents,  # legacy supports text; file mixing may differ
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",
                },
            ).text

    def _retry_wrapper(self, fn, *args, **kwargs):
        err_last = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                transient = ("503" in msg) or ("UNAVAILABLE" in msg) or ("RESOURCE_EXHAUSTED" in msg)
                if not transient and attempt >= MAX_RETRIES:
                    raise
                if transient:
                    delay = BASE_DELAY * (2 ** (attempt - 1))
                    delay = delay * (0.7 + 0.6 * random.random())
                    time.sleep(min(delay, 15.0))
                    err_last = e
                    continue
                err_last = e
                break
        if err_last:
            raise err_last

    def generate_json(self, prompt: str, system_instruction: str = None, temperature: float = 0.2, max_output_tokens: int = 20000):
        contents = (system_instruction + "\n\n" + prompt) if system_instruction else prompt
        try:
            return self._retry_wrapper(self._call_once, self.model_name, contents, system_instruction=None, temperature=temperature, max_output_tokens=max_output_tokens)
        except Exception:
            if FALLBACK_MODEL and FALLBACK_MODEL != self.model_name:
                return self._call_once(FALLBACK_MODEL, contents, system_instruction=None, temperature=temperature, max_output_tokens=max_output_tokens)
            raise

    def generate_json_with_file(
        self,
        file_path: str,
        instruction: str,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 20000,
    ):
        if self.mode != "new":
            # Legacy SDK fallback: concatenate system + instruction as plain text
            merged = (system_instruction + "\n\n" + instruction) if system_instruction else instruction
            return self.generate_json(
                merged,
                system_instruction=None,  # already merged
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

        def call(model):
            file_obj = self.client.files.upload(file=file_path)
            contents = []
            if system_instruction:
                contents.append(system_instruction)
            contents.append(instruction)
            contents.append(file_obj)
            return self.client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",
                },
            ).text

        try:
            return self._with_retry(lambda: call(self.model_name))
        except Exception:
            from os import getenv
            fallback = getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")
            if fallback and fallback != self.model_name:
                return call(fallback)
            raise
