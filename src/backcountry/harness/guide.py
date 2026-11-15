"""`backcountry guide`: how an agent runs backcountry for someone, shipped with
the package so it matches the installed version."""

GUIDE = """\
How to run backcountry for someone (an agent's guide; a person can follow it too)

1. Ask what to make faster: the function their app calls for one request
   (write a reply, make an image, classify a batch). Add two lines to their
   code:
     import backcountry
     @backcountry.tune            # on that function, where it is defined
   It must take plain data (text, numbers, arrays, files, the model) and
   return the request's result: the search replays it from the arguments the
   app passed and compares what it returns. If the app's function only prints
   its result, or is handed live parts of the running app (a callback, a
   queue, a server's event loop), move the request's work into a function
   that takes the input and returns the output, and mark that; say so to the
   user. The script must run an MLX model and make the same calls each run (its
   arguments go after --). It need not call the function: then
     backcountry init app.py --questions
   lists its arguments instead (a server never ends: its loop goes under
   `if __name__ == "__main__":`, and --ask leaves that block out). Ask the user
   for one request in their words and give it with
   --request "question=...; history=chat.json". Otherwise --questions runs the
   app once and lists the sizes that change the model's work (a prompt's
   tokens, a batch's count, an image's width and height, a number argument),
   numbered, and whether the search may rewrite the function itself (--keep
   same-output, the default) or only make the model's math faster (same-math).
   A Hugging Face id or local folder MLX-LM loads needs no script.
   Ask the user each question in plain words, offering the default (the sizes
   the app called with).
2. Write their answers:
     backcountry init app.py --yes --sizes "1=32-4096; 2=256" --keep ... --attempts ...
   (sizes by question number: a list 1, 8, 32, a range 32-4096 or pairs 512x512)
   (a language model: --optimize prompts|replies|chat, --prompt-tokens ...)
   It prints the plan in a sentence: read it back to the user.
3. Check (no attempts; minutes, up to an hour when a request generates
   thousands of tokens, since it records every GPU call at each size):
     backcountry check backcountry.yaml
   Show its summary: the AI is signed in, what fits in memory, the room the
   hardware leaves (headroom), and its verdict. If it says "may not be worth a run",
   say so and ask before going on.
4. On a yes, start it in the background (it survives this session closing):
     backcountry start backcountry.yaml
5. Loop: run `backcountry status --wait` (add --json for fields) and relay
   each milestone it returns at: a kernel shipped, a region opened or closed,
   a pause, resume or crash, the final result. Say nothing in between.
6. Controls, on the user's word: backcountry pause | resume --background |
   finish | skip (the newest run; --work-dir names another). finish still runs
   the final checks and packages the result.
7. At the end, report the confirmed speedup and where it holds, whether the
   outputs are identical, and the artifact path; or say plainly that nothing
   beat the baseline, which is a normal result. A marked function's result
   is written beside the file that defines it, in
   backcountry-results/<function>/, and the function uses it from its next
   call on: commit or ship that folder with the code. To run untouched:
   BACKCOUNTRY_OFF=1, or @backcountry.tune(apply=False); to remove it,
   delete the folder. A model id's result is used with:
     from backcountry import apply      # model = apply(model)
     from backcountry import generate   # a faster generation loop, MLX-LM's arguments

Rules: one job at a time on a Mac; keep other GPU work quiet while it runs;
never edit a run's files while it runs; pause before the Mac sleeps.
"""


def main() -> int:
    print(GUIDE, end="")
    return 0
