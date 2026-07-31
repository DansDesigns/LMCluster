"""Talking to the model.

There is exactly one inference path in LMCluster: the llama-server this
node launched, which is holding a model whose layers may be spread across
several machines. Whether it is sharded or sitting entirely on this one
machine makes no difference here — a shard with no remote workers is just
an ordinary local llama-server, so the same code covers both.

This used to be a set of interchangeable backends, so that a different
program could be running a different model on each machine. That is the
opposite of what LMCluster does now, so it has gone, and what is left is a
thin client over llama-server's OpenAI-compatible endpoints.
"""

import json
import time

import httpx


class NoModelLoaded(RuntimeError):
    """Raised when something asks for inference and nothing is loaded."""


class Engine:
    def __init__(self, master, defaults: dict | None = None):
        self.master = master        # rpc.ShardMaster
        # Saved preferences from the settings page. A single request can
        # override any of them; leaving a field out of the request means
        # "use whatever is saved", which is what makes the settings page
        # meaningful rather than decorative.
        self.defaults = defaults if defaults is not None else {}

    def _setting(self, name, override, fallback):
        if override is not None:
            return override
        value = self.defaults.get(name)
        return fallback if value is None else value

    # -- state ------------------------------------------------------------

    def url(self) -> str | None:
        status = self.master.status()
        return status["url"] if status["running"] else None

    def _require_url(self) -> str:
        url = self.url()
        if url is None:
            raise NoModelLoaded(
                "no model is loaded. Load one from the Pool page first.")
        return url

    async def info(self) -> dict:
        """What is loaded, and across how many machines."""
        url = self.url()
        if url is None:
            return {"loaded": False, "model": None, "nodes": 0, "workers": []}
        plan = self.master.plan or {}
        model = None
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{url}/v1/models")
                r.raise_for_status()
                data = r.json().get("data", [])
            if data:
                model = data[0].get("id")
        except (httpx.HTTPError, ValueError, KeyError):
            # The server is up but not answering yet; report the plan we
            # know about rather than claiming nothing is loaded.
            pass
        return {
            "loaded": True,
            "model": model or plan.get("model_path", "").split("/")[-1],
            "nodes": len(plan.get("workers", [])) + 1,
            "workers": [w["name"] for w in plan.get("workers", [])],
            "model_size": plan.get("model_size"),
            "ctx": plan.get("ctx"),
        }

    # -- inference --------------------------------------------------------

    def _messages(self, prompt: str, system: str | None,
                  history: list | None) -> list:
        messages = []
        # An empty string is a deliberate "no system prompt", so only fall
        # back to the saved one when the request says nothing at all.
        if system is None:
            system = self.defaults.get("system_prompt") or None
        if system:
            messages.append({"role": "system", "content": system})
        for turn in (history or []):
            role = turn.get("role")
            if role in ("user", "assistant") and turn.get("content"):
                messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _sampling(self, temperature=None, max_tokens=None) -> dict:
        """Sampling options, saved defaults filled in where not given."""
        body = {
            "temperature": self._setting("temperature", temperature, 0.7),
            "top_p": self._setting("top_p", None, 0.95),
            "top_k": self._setting("top_k", None, 40),
            "min_p": self._setting("min_p", None, 0.05),
            "repeat_penalty": self._setting("repeat_penalty", None, 1.1),
        }
        limit = self._setting("max_tokens", max_tokens, 0)
        if limit:
            body["max_tokens"] = limit
        return body

    async def generate(self, prompt: str, system: str | None = None,
                       history: list | None = None,
                       temperature: float | None = None,
                       max_tokens: int | None = None) -> dict:
        url = self._require_url()
        body = {"messages": self._messages(prompt, system, history),
                **self._sampling(temperature, max_tokens)}
        started = time.time()
        # Very long timeout on purpose: a large model spread over a LAN can
        # take minutes for one answer, and dying halfway through is worse
        # than waiting.
        async with httpx.AsyncClient(timeout=3600) as client:
            r = await client.post(f"{url}/v1/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        return {
            "text": text,
            "seconds": round(time.time() - started, 2),
            "tokens": usage.get("completion_tokens"),
        }

    async def stream(self, prompt: str, system: str | None = None,
                     history: list | None = None,
                     temperature: float | None = None,
                     max_tokens: int | None = None):
        """Yield {"delta": str} as tokens arrive, then a final summary dict.

        Streaming matters more here than it would elsewhere. A model spread
        across four machines may produce a couple of tokens a second, so
        without streaming the interface would simply appear frozen for
        minutes at a time.
        """
        url = self._require_url()
        body = {"messages": self._messages(prompt, system, history),
                "stream": True,
                **self._sampling(temperature, max_tokens)}

        started = time.time()
        first_token_at = None
        collected = []

        async with httpx.AsyncClient(timeout=3600) as client:
            async with client.stream("POST", f"{url}/v1/chat/completions",
                                     json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    delta = (chunk.get("choices", [{}])[0]
                             .get("delta", {}).get("content"))
                    if delta:
                        if first_token_at is None:
                            first_token_at = time.time()
                        collected.append(delta)
                        yield {"delta": delta}

        elapsed = time.time() - started
        count = len("".join(collected).split())
        yield {
            "done": True,
            "text": "".join(collected).strip(),
            "seconds": round(elapsed, 2),
            # Time to first token is the figure that tells you whether the
            # cluster is healthy: it is dominated by weight loading and
            # network setup, so a sudden jump usually means a slow link.
            "first_token": (round(first_token_at - started, 2)
                            if first_token_at else None),
            "words_per_second": (round(count / elapsed, 2)
                                 if elapsed > 0 and count else None),
        }


# -- skill generation ----------------------------------------------------

SKILL_SYSTEM = """You write small, self-contained Python modules called \
skills for a program called LMCluster.

A skill is one Python file. It must define exactly one entry point:

    def run(inputs):
        ...
        return {...}

`inputs` is a dict. The return value must be a dict of plain JSON types \
(strings, numbers, booleans, lists, dicts, None).

The file must begin with a docstring in exactly this format:

    \"\"\"
    Skill: A Short Title
    Description: One sentence saying what it does.
    Version: 1.0
    Author: cluster
    Inputs: {"field_name": "string"}
    Outputs: {"field_name": "number"}
    Tags: comma, separated
    \"\"\"

The Inputs and Outputs lines must be valid JSON objects on a single line, \
mapping field names to one of: string, number, boolean, list, dict.

Rules:
- Standard library only unless the request clearly needs otherwise.
- No network access, no subprocesses, no deleting files.
- Handle missing or malformed inputs by returning an error field rather \
than raising.
- Keep it short and readable.

Reply with the Python file and nothing else. No explanation, no markdown \
fences."""


def strip_fences(text: str) -> str:
    """Remove markdown code fences a model may have added anyway.

    The system prompt asks for bare Python, but models add fences often
    enough that silently handling it is kinder than failing validation and
    making the user retry.
    """
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]  # drop the opening fence and any language tag
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


async def generate_skill(engine: Engine, description: str,
                         temperature: float = 0.3) -> dict:
    """Ask the loaded model to write a skill, then check it compiles.

    Returns the source for review rather than saving it. Nothing written by
    a model should land on disk as executable code without someone reading
    it first, and skills are not sandboxed.
    """
    from . import skills

    prompt = f"Write a skill that does the following:\n\n{description}"
    # Deliberately overriding the saved settings here. Code wants a low
    # temperature whatever the conversation preference is, and the user's
    # own system prompt would be actively unhelpful.
    result = await engine.generate(prompt, system=SKILL_SYSTEM,
                                   temperature=temperature)
    source = strip_fences(result["text"])
    valid, problems = skills.validate(source)
    meta = skills.parse_metadata(source)
    return {
        "source": source,
        "valid": valid,
        "problems": problems,
        "name": meta.get("name"),
        "description": meta.get("description"),
        "inputs": meta.get("inputs"),
        "outputs": meta.get("outputs"),
        "seconds": result["seconds"],
    }
