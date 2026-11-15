# backcountry

Make an MLX model faster on Apple Silicon without changing its answers.

You mark the function your app calls for one request. An AI proposes faster
versions: custom Metal GPU code (kernels) for parts of the model and, if you
allow it, a faster way to use the model (a reply drafted several tokens at a
time and checked). A separate harness tests every one, and keeps it only if
the outputs still match the original (bit for bit, or within a tolerance you
control when the math is reordered) and the **whole request** gets measurably
faster. Whatever survives is saved beside your code, and your function uses it
from its next call.

Sometimes nothing survives. That's a real answer, not a crash: no candidate
beat your code by a margin the tool could confirm within its attempt budget.

## Quickstart

You need an Apple Silicon Mac and Python 3.12, 3.13 or 3.14. Python 3.15 isn't
supported yet (MLX's build for it doesn't let backcountry record the model); a
new uv project may pick 3.15 by default, so pin one first: `uv python pin 3.14`.

1. **Install [Claude Code](https://code.claude.com/docs/en/setup)**, run
   `claude` once, sign in, and exit. Claude Code is the AI that writes the
   kernels, so your account needs Claude Code access. A signed-in Codex or
   Gemini CLI works too, or the Anthropic API with `ANTHROPIC_API_KEY`.
2. **Install backcountry** in the environment your app runs in:

```sh
pip install backcountry        # or: uv add backcountry
```

3. **Mark the function** your app calls for one request, the one that turns
   an input into an output (writes a reply, makes an image, classifies a
   batch), with two lines:

```python
import backcountry

@backcountry.tune
def generate_text(model, tokenizer, prompt, max_tokens=150):   # your function, unchanged
    ...
```

4. **Run four commands** from your project's folder:

```sh
backcountry init app.py            # runs app.py once; asks the sizes to optimize
backcountry check backcountry.yaml # no attempts: does it run, fit, have room? (minutes; longer for long replies)
backcountry start backcountry.yaml # the search, in the background
backcountry status --wait          # returns at each milestone
```

A few things to know while it runs:

- **Expect it to take hours.** Each attempt is checked and timed, and the
  budget counts attempts, not minutes. `start` runs in the background, so you
  can close the terminal.
- **Keep other GPU work quiet.** Background load makes the timings noisy.
- **Kernel requests use your AI provider's allowance** (subscription or API
  credits).

When it finishes, the result is a folder beside the file that defines your
function, `backcountry-results/<function>/`. Commit it, or ship it with your
code: the function uses it from its next call, and says so once.

## Run it through an AI agent

Don't want to answer questions and watch logs? Install backcountry, then paste
this into a coding agent (Claude Code, Codex, …) opened on your project:

```text
Run `backcountry guide` and follow it to make my app faster.
Ask me what you need, and update me only at milestones.
```

`backcountry guide` prints the steps for the installed version: where to put
the mark, which questions to ask you, how to start and watch the run. There
are two AIs here, and they can come from different providers: the agent you're
chatting with runs and watches the job; the one named by `--judge` writes the
kernels.

## What init asks

The function can do whatever one request does in your app: tokenize, run
your own loop with your own cache, sample, stop at an end token, decode,
post-process, print. It has to return the request's result (the search
compares what it returns) and take plain data: text, numbers, arrays, files,
the model. A function that only prints its result, or that is handed live
parts of the running app (a callback, a queue, a server's event loop), can't
be replayed; move the request's work into a function that takes the input
and returns the output, and mark that. `init` runs `app.py` as you would (`python app.py`) and
records each call of the marked function and the model it uses (passed in,
held by an object it is given, or a global, even one a library keeps for
itself). Then it tries each argument of the first call at another size and
watches what the model does, so it asks only about the ones that change the
work:

```
These change how much work the model does:
  1. width and height (together)
These run the model more or fewer times (one value, used for every size):
  2. steps
These don't change the work:
  prompt, seed
Sizes for width and height (together): a list like 512x512, 1024x768 [default: 1024x1024]: 512x512, 1024x1024
steps: the value to use for every size [default: 4]:
```

The same questions come for any model: a prompt's length in tokens, a batch's
count, an array's axis, an image's width and height, any number argument.
Each question's default is the sizes your app called it with (here the
script made one 1024x1024 image), so pressing Enter optimizes what the app
actually runs. Answer with a list (`1, 8, 32`), a range (`32-4096`, spread into five sizes)
or pairs (`512x512, 1024x768`); at most eight sizes in all, since each is
checked and timed. `init` makes the call at every size once before writing
anything; agents pass the answers by number (`--sizes "1=512x512, 1024x1024; 2=4"`).
The search replays your function at each size, from the arguments your app
passed, repeated or cut to the size (never made-up data); it compares what the
function returns (arrays, numbers, text by its bytes, images) and times the
whole call, so the speedup it reports is your request's. Random draws are
seeded the same for every replay, and what the function prints is muted.

`init` also asks whether the search may rewrite the function itself
(`same-output`, the default). The AI then first proposes new versions of it,
with the same signature, that use the model differently: a loop that does
less work, a reply drafted several tokens at a time and checked. One ships
only if everything it returns matches the original's on fresh inputs and the
whole call is faster. The kernel search follows. `same-math` keeps your
function exactly as written and only makes the model's math faster.

A serving engine is marked the same way, whatever engine it is: mark the
function that handles one request (an engine's generate call, a server's route
handler). The engine may run the model on worker threads of its own, on a
stream of its own, and keep the model on its own objects (`engine.model`
beside your global): the search replays the request the same way, puts its
copies of the model everywhere the app holds it, and may rewrite the handler
and the code it calls.

An app that makes no request itself is asked for one. When the script ends
without calling the marked function, `init` asks for each of its arguments
(text, a number, a list, or a file beside the app: an image, a `.json`
conversation, a `.txt`); an argument named like one of the app's globals (its
`model`, its `tokenizer`) is taken from there. A server never ends: put its
serving loop under `if __name__ == "__main__":` and run
`backcountry init app.py --ask`, which loads the app without that block and
asks. One typed request is enough: the size questions repeat or cut it to
each size. An agent reads the questions with `--questions` and answers with
`--request "question=What is MLX?; history=chat.json"`.

What the app must do: run an MLX model (a PyTorch script is told so before it
runs), make the same calls each time it runs (`init` and the search rerun it),
and give the same output, doing the same work, for the same input (seeded randomness is fine; a serving engine's prompt cache, which skips reading a request it has just read, is not: `check` says so). The
app's own arguments go after `--` (`backcountry init app.py -- --image cat.png`).

`backcountry.yaml` holds what you chose: the sizes (one line each), a request
you typed, whether the function may be rewritten, the attempts. Edit it and
the next `check` or `start` uses it. Its paths are read from its own folder, so
a project copied or cloned to another Mac runs as it is: commit
`backcountry.yaml` and `.backcountry/backcountry_model.py` with your code (the
rest of `.backcountry/`, this machine's recording, is git-ignored). `check`
says how much headroom the hardware leaves at each size ("up to 14%
headroom"; under 5%: "may not be worth a run"), what fits in memory, and that
the AI is signed in.

## While it runs

The commands below act on the newest run under `runs/` in your project;
`--work-dir` names another.

```sh
backcountry status          # one screen; --json for scripts
backcountry status --wait   # returns at the next milestone: a kernel shipped, a region
                            # opened or closed, a pause or crash, the final result
backcountry finish          # stop new attempts; still runs the final checks and saves the result
backcountry skip            # leave a part of the model that isn't going anywhere
backcountry pause           # need the Mac back (sleep, restart, other GPU work)?
backcountry resume          # pick it up where it stopped
```

`--attempts N` (at `init`, or in `backcountry.yaml`) is how many proposals the
search may try (up to 25 on any one part of the model). There is no time
limit: an attempt takes minutes on a small model and up to half an hour on a
large one, and the run ends when its attempts are spent (or at
`backcountry finish`). After the search, each kernel is checked at every size
in the range, then timed and saved: that is extra time. The result says how
many attempts it rests on and how long each part took.

Each run's folder holds `report.json` (every measurement, attempt and failure,
plus the final outcome), `run.jsonl` and `candidates.log` (progress and a
result for each attempt). The final summary says which of three things
happened: a verified result, no confirmed improvement, or a failure. A kernel
that is fast on its own doesn't count; only a confirmed whole-request speedup
ships.

## Using the result

A marked function needs nothing more: its first call installs the kernels
into the model it uses (and, when the search found a faster version of the
function, calls that from then on). A faster version is made for the
function's code as the run saw it: once you edit the function, the app runs
your edited code (it says so; the kernels still apply) until a new run. To run
it untouched, set `BACKCOUNTRY_OFF=1`, or keep the mark with
`@backcountry.tune(apply=False)`; to remove the result, delete its folder.

Elsewhere, install a result into a model you hold:

```python
from backcountry import apply
model = apply(model)   # always use the returned model
```

Everything after that line (generation, batching, caches) stays as it was.
A finished run also copies its result to `~/.backcountry/artifacts/` (set
`BACKCOUNTRY_HOME` to move it), and `apply(model)` uses the newest one there
whose layers your model has; a model none fits runs as it is, with a warning.
To use a particular result, name its folder: `apply(model, "path/to/artifact")`.

**See it yourself, through your own code.** `compare` calls your function with
the model as loaded and with the optimization switched in, alternately, on one
copy of the weights, and tells you whether the outputs are identical and how
much faster it is:

```python
from backcountry import compare
print(compare(lambda m: generate(m, tokenizer, prompt=my_prompt, max_tokens=100), model))
# 48 layers replaced; outputs identical; 1.124x faster (as loaded 1.503 s, optimized 1.337 s, ...)
```

Keep in mind:

- **Weights aren't included.** You load them yourself as usual. The model's
  architecture and quantization have to match what was optimized, but the
  weight values can differ.
- **Only tested input sizes get the new kernels.** Anything else quietly runs
  the original code, so it's still correct, just not faster.
- **Each layer is checked first.** A layer that isn't the one the kernels were
  made for (another model, quantization or MLX version) keeps its own
  code, with a warning.
- **Speed is measured on the run's chip.** On another Mac the outputs are the
  same, and `apply()` warns that the speed there is unmeasured.

More in [the result guide](https://github.com/shivamg05/backcountry/blob/main/docs/artifacts.md).

## Other ways to run it

A language model MLX-LM loads can be named directly, with no app:

```sh
backcountry init mlx-community/Qwen3-8B-4bit
```

`init` then asks whether prompts, replies or both should get faster, the
sizes, whether tokens may come from a faster generation loop (`same-output`)
or only faster kernels (`same-math`), and how many attempts to make. Use the
result through MLX-LM's own functions, imported from backcountry: they take
the same arguments and give the same results, and use the loop stored for the
model you pass (any other model runs MLX-LM's own):

```python
from backcountry import apply, generate   # or stream_generate, generate_step
model = apply(model)
text = generate(model, tokenizer, prompt, max_tokens=256)
```

For full control (several kinds of call, your own limits, a size range), write
a model file and a manifest yourself: see the [usage guide](https://github.com/shivamg05/backcountry/blob/main/docs/usage.md) and
[write a manifest](https://github.com/shivamg05/backcountry/blob/main/docs/manifest.md). Any supported AI can write the kernels:
`--judge codex --model YOUR_MODEL_ID`, for example.

## Further reading

- [Limitations](https://github.com/shivamg05/backcountry/blob/main/docs/limitations.md): what a result does and doesn't promise.
  Worth reading before you quote a number.
- [RUNNING.md](https://github.com/shivamg05/backcountry/blob/main/RUNNING.md): where an agent running a job for you should start.
- [Architecture](https://github.com/shivamg05/backcountry/blob/main/docs/architecture.md): how the harness works and the rules it never breaks.
- [Judge protocol](https://github.com/shivamg05/backcountry/blob/main/docs/judge-protocol.md): how to plug in another AI provider,
  or answer kernel requests by hand.

## License

MIT.
