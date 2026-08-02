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
import os
import re
import time

import httpx


class NoModelLoaded(RuntimeError):
    """Raised when something asks for inference and nothing is loaded."""


class ModelLoading(RuntimeError):
    """Raised when a model is on its way in but not yet usable.

    Worth its own exception rather than being folded in with a lost
    connection. llama.cpp opens its port the moment it starts and answers
    every request with 503 until the weights are actually in memory, which
    on a large model spread over a network is minutes. Reporting that as
    lost contact sends somebody hunting for a machine that has dropped off
    when in fact everything is fine and they simply asked too early.
    """


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

    async def state(self) -> str:
        """One of: none, loading, ready, unreachable.

        llama.cpp answers /health with 200 once the model is usable and 503
        while it is still reading it in. That distinction matters more here
        than it would elsewhere, because a large model spread across
        several machines takes minutes to load while its port is open the
        whole time, and asking too early gets a 503 that looks alarming and
        is not.

        A build without a health endpoint is treated as ready rather than
        broken. It answered, which is the important part, and refusing to
        talk to a working server because it lacks one endpoint would be a
        worse failure than the one being guarded against.
        """
        url = self.url()
        if url is None:
            return "none"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{url}/health")
        except httpx.HTTPError:
            return "unreachable"

        if r.status_code == 200:
            return "ready"
        if r.status_code == 503:
            return "loading"
        if r.status_code == 404:
            return "ready"          # older build, no health endpoint

        # Anything else: ask whether it can list models, which every build
        # can do, before concluding it is not answering at all.
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r2 = await client.get(f"{url}/v1/models")
            return "ready" if r2.status_code == 200 else "unreachable"
        except httpx.HTTPError:
            return "unreachable"

    async def require_ready(self) -> str:
        url = self._require_url()
        state = await self.state()
        if state == "loading":
            waited = self.master.status().get("uptime")
            raise ModelLoading(
                "The model is still loading"
                + (f" — {waited}s so far" if waited else "")
                + ". A large model spread over several machines takes a "
                  "while to read in. Ask again in a minute; the Pool page "
                  "shows when it is ready.")
        if state == "unreachable":
            raise RuntimeError(
                "llama.cpp is running but not answering. Check the load log "
                "on the Pool page.")
        return url

    async def info(self) -> dict:
        """What is loaded, and across how many machines."""
        url = self.url()
        if url is None:
            return {"loaded": False, "state": "none", "model": None,
                    "nodes": 0, "workers": [],
                    # If it died rather than never having been started, say
                    # so. "No model loaded" on its own is true and useless.
                    "failure": self.master.status().get("last_failure")}

        state = await self.state()
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
            "loaded": state == "ready",
            "state": state,
            "loading_for": self.master.status().get("uptime"),
            "progress": self.master.progress(),
            "model": model or os.path.basename(
                plan.get("model_path", "").replace("\\", "/")),
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
        url = await self.require_ready()
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
        url = await self.require_ready()
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
- Plain ASCII punctuation only. No curly quotes, no em dashes, no ellipsis \
characters. Use ' and " and - and ...

Reply with the Python file and nothing else. No explanation before it, no \
commentary after it, no markdown fences."""


# Characters models reach for when writing prose, which are not valid
# Python outside a string. An em dash in the first column stops the file
# compiling with "invalid character (U+2014)", and a curly quote in place
# of an apostrophe does the same. Substituting the plain equivalents is
# safe: in code they are always mistakes, and in a docstring the plain
# version reads no worse.
TYPOGRAPHIC = {
    "\u2014": "-",   # em dash
    "\u2013": "-",   # en dash
    "\u2212": "-",   # minus sign
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u00a0": " ",   # non-breaking space
    "\u2026": "...",  # ellipsis
    "\u00d7": "*",   # multiplication sign
    "\u2192": "->",  # arrow
}


def tidy_characters(text: str) -> str:
    for wrong, right in TYPOGRAPHIC.items():
        text = text.replace(wrong, right)
    return text


_FENCED = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)
_FENCE_OPEN = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*)", re.S)
_CODE_START = ('"""', "'''", "import ", "from ", "def ", "class ", "#!")


def extract_code(text: str) -> str:
    """Pull the Python out of a model's reply.

    The instructions ask for bare Python and nothing else. Models comply
    most of the time and then, occasionally, open with "Here is the skill
    you asked for:" or a stray dash, wrap the code in a fence, and sign off
    with "Hope that helps!". The previous version only removed a fence
    sitting at the very start of the reply, so any of that left the prose
    in place and the file would not compile — which is exactly how a
    generation that took eighty-six seconds came back reporting a syntax
    error on line one.
    """
    text = tidy_characters(text.strip())

    # A fenced block anywhere in the reply is the clearest signal.
    match = _FENCED.search(text)
    if match:
        return match.group(1).strip()

    # A fence that was opened and never closed, which happens when a reply
    # is cut short.
    match = _FENCE_OPEN.search(text)
    if match:
        return match.group(1).strip()

    # No fence: drop everything before the first line that could plausibly
    # begin a Python file.
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith(_CODE_START):
            return "\n".join(lines[i:]).strip()

    return text


# Kept under the old name because it is the obvious thing to reach for.
strip_fences = extract_code


def describe_result(source: str, seconds: float, attempts: int) -> dict:
    from . import skills
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
        "seconds": round(seconds, 1),
        "attempts": attempts,
    }


def retry_prompt(description: str, source: str, problems: list) -> str:
    """Hand the model its own mistake rather than handing it to the person.

    Waiting a minute and a half for a skill and being told it will not
    compile is a poor result when the fault is usually a missing function
    or a stray character, and the model can fix either if simply told what
    it did.
    """
    faults = "\n".join(f"- {p}" for p in problems)
    return (f"That did not work. The file you produced has these "
            f"problems:\n\n{faults}\n\nHere is what you sent:\n\n"
            f"{source[:2000]}\n\nWrite the whole file again, corrected. "
            f"The original request was:\n\n{description}")


async def generate_skill(engine: Engine, description: str,
                         temperature: float = 0.3,
                         on_delta=None, retries: int = 1) -> dict:
    """Ask the loaded model to write a skill, then check it compiles.

    Returns the source for review rather than saving it. Nothing a model
    writes should land on disk as runnable code without a person reading it
    first, and skills are not sandboxed.

    If `on_delta` is given it is called with each fragment as it arrives, so
    the caller can show the model working rather than a spinner. A model
    spread across several machines can take a couple of minutes over this,
    and watching it is the more useful arrangement: when the result is
    wrong, you have already seen where it went wrong rather than being
    handed a verdict about a file you never saw being written.
    """
    started = time.time()
    prompt = f"Write a skill that does the following:\n\n{description}"
    source = ""

    for attempt in range(retries + 1):
        collected = []
        # Deliberately ignoring the saved chat settings. Code wants a low
        # temperature whatever the conversation preference is, and the
        # person's own system prompt would be actively unhelpful here.
        async for chunk in engine.stream(prompt, system=SKILL_SYSTEM,
                                         temperature=temperature):
            if chunk.get("delta"):
                collected.append(chunk["delta"])
                if on_delta:
                    await on_delta(chunk["delta"])

        source = extract_code("".join(collected))
        result = describe_result(source, time.time() - started, attempt + 1)
        if result["valid"] or attempt == retries:
            return result

        problems = [p for p in result["problems"]
                    if not p.startswith("warning:")]
        if on_delta:
            await on_delta(f"\n\n--- that will not compile: "
                           f"{'; '.join(problems)}\n--- asking again ---\n\n")
        prompt = retry_prompt(description, source, problems)

    return describe_result(source, time.time() - started, retries + 1)
