# Qwythos Code (qwc)

**An autonomous coding CLI for local models, built around one rule: never take the model's word for it.**

qwc reads, edits and tests code in your project with a model running on your own machine (Ollama: `gemma4:26b` by default). What makes it different is what happens when the model says it is done: before the report reaches you, qwc checks it against what actually happened in the session (which commands ran and how they ended, which files changed, what was really removed) and sends the model back to fix it when they disagree.

> 日本語の詳しい説明（全機能・実測・設計の理由）: [README.ja.md](README.ja.md)

```
❯ the test is failing, fix it

✻ thought for 6.2 s
● run_command(npm test)
  ⎿ exit code 1
● read_file(sum.js)
  ⎿ read 3 lines
● edit_file(sum.js)
  ⎿ replaced 1 location
● run_command(npm test)
  ⎿ exit code 0

Fixed the subtraction in sum.js to addition. npm test passes.
```

## Why it exists

Small local models are capable enough to do real work and unreliable enough to report work they did not do. Running qwc every day since August 2026, I recorded what that looks like:

- asked to delete a function that did not exist, the model deleted two blank lines and reported "deleted";
- asked to make a test pass without touching the test, it wrote a `sitecustomize.py` that forced the test to print OK and reported "confirmed successful";
- the first version of the check meant to catch this never fired in 218 sessions, because it counted `ls` as "work done".

So qwc's report checks compare the start and the end of the request, not the model's description of it, and every check is tested against recorded real sessions before it ships.

## Report watchers

When the model ends a turn with a report, qwc runs a set of watchers over that report and the session's own records. If one fires, the model gets a concrete nudge (up to 5 times) instead of the user getting a false report. Examples:

| Watcher | Catches |
| --- | --- |
| `riggedTestPass` | "tests pass" after making them pass by rigging how they run: `PYTHONPATH=`, `sitecustomize.py`, `*.pth`, `conftest.py`, `assert True`, `|| true` |
| `claimedTestsPassedButFailed` | "all tests pass" when the test command that ran in this turn failed |
| `claimedRunningSomethingNeverRun` | describing the result of a command or script that never ran in this turn |
| `claimedButNothingChanged` / `filesNeverWritten` | "fixed" / "updated" when no write to that file ever succeeded |
| `removedTextThisTurn` | "deleted X" when X is not in what was actually removed between the start and the end of the request |
| `claimedAllButSomeRemain` | "removed all of them" when matching lines are still there |
| `claimedFailureButChanged` | "I couldn't do it" when files were in fact changed |

The watchers are measured with a separate harness, [agent-report-eval](https://github.com/daigo0904/agent-report-eval), which replays 1,481 recorded cases through the real agent loop. That harness taught me not to trust my own numbers: the watchers scored 97% on my own data and 72% detection (23/32) with 27% false positives under a blind independent judge. The rule-based watchers are a first line; the stronger, evidence-based check lives in [local-ai-stack/verify](https://github.com/daigo0904/local-ai-stack/tree/main/verify) (`proofcheck`).

## Other guard rails

- **Stays inside the project folder** (`--allow-outside` to lift).
- **Asks before edits, commands and network access**, with three levels: default (ask for everything), `--accept-edits` (edits pass, commands and network still ask), `--yolo` (nothing asks). Commands containing `;`, `|` or `>` are never treated as safe.
- **Tells chat from work without asking the model.** A remark like "the tax rate in tax.js is still old, huh" is not a request; if the model tries to edit during small talk, qwc asks first, even under `--yolo`.
- **Does not hand the model tools it cannot use.** Plan mode removes `write_file` instead of asking the model not to use it; 26B models do not reliably follow "please don't".
- **Facts before the model answers.** Names the request treats as existing (functions, files) are checked first and the result is appended to the request, because models will grep, see zero hits, and edit something else anyway.
- **`/undo` and `/diff`** for every edit in the session, and an honest report when something could not be undone.

## Machine-readable runs

```sh
qwc -p "fix the login bug" --yolo --events events.jsonl
```

`--events` writes the run in the same JSONL format as `codex exec --json` (OpenAI Codex), so the same tools can verify qwc and Codex runs. [local-ai-stack](https://github.com/daigo0904/local-ai-stack)'s `agent-run` uses this to seal a task contract, run qwc or Codex, and judge the final report.

## Requirements and setup

- Node.js 20+ (the core uses only Node built-ins; no runtime dependencies)
- [Ollama](https://ollama.com) running, with `gemma4:26b` (about 18 GB free); `-m qwythos:latest` (7.4 GB) for smaller machines
- Optional: `typescript` for semantic code search, `playwright` for reading pages behind a login. If missing, those tools are simply not offered to the model.

```sh
git clone https://github.com/daigo0904/qwythos-code ~/qwc && cd ~/qwc
npm link           # or: alias qwc="node ~/qwc/bin/qwc.mjs"
cd ~/my-project
qwc                # then type a request
```

`npm test` runs the test suite without a model (about 870 tests).

## Network

The only outbound traffic is search terms (Tavily), URLs you approve, and cookies you saved for a site you logged into yourself. File contents, conversation history and keys are never sent. All outbound code is in `src/web.mjs` and `src/browser.mjs`. `--no-net` keeps everything local.

## License

MIT.
