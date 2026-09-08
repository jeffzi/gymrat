# Vision

gymrat runs an optimization loop under an AI agent: the agent edits, gymrat measures the edit
against the baseline with paired statistics, and only a defensible improvement is kept. It competes
with telling an agent "make it faster" and trusting the number it reports back. The same loop runs
by hand, and `compare` and `measure` expose the same engine for a one-off verdict in CI or a code
review.

Metrics jitter from run to run. A benchmark that prints 3% faster is often noise, and a pass rate
over a task corpus moves between runs of the same code. An agent that keeps whatever number went
down ratchets on noise, faster than a person judging by eye would. More runs do not fix this.
Pairing the samples, testing the paired differences, and refusing any verdict the data cannot
support does.

The pillars below decide design questions. When two pull apart, the lower number wins, and the
change names the conflict instead of resolving it silently. The vetoes under "What gymrat is not"
outrank every pillar. Throughout, the caller is the person or the AI agent driving gymrat. The
commands and verdict names used below are defined in [the README](README.md).

## Pillars

### 1. Evidence beats impressions

gymrat would rather say nothing than something it cannot defend. `no-signal` beats a false
improvement, `unstable` beats a guess, and a verdict without paired samples is not a verdict.
gymrat refuses an iteration that no longer fits the time available. It never shrinks one to fit.

### 2. gymrat owns setup and bookkeeping

Anything with no judgment in it is setup or bookkeeping, and gymrat owns it. The caller owns edits
and decisions. Opening a session, recording a baseline, tracking time, advancing the baseline after
a keep, refusing an action the log already shows to be wrong: each has one correct outcome. gymrat
does them. Choosing what to edit, reading a verdict, deciding whether a stepping-stone change
is worth keeping: these need judgment, so gymrat presents the evidence and leaves the call to the
caller.

A rule that a prompt has to remind the agent of is a rule gymrat has not enforced yet. A
judgment-free rule belongs in code: a command that refuses, a value the output prints, or a check
that runs before anything expensive. gymrat enforces a rule that is judgment-free except for a rare
case, and names an option for the exception. Prose covers only the decisions that remain.

### 3. Automation never destroys a measured improvement

A measured improvement is an iteration that improved its primary metric and cleared every gate. An
automated step may keep one or leave it for a person to decide. It never reverts one. Automation
settles an iteration only the way the `keep` gate would settle it unattended. When it cannot tell,
it leaves the work in place and blocks the next run until a person decides. It never guesses.

### 4. Humans and agents, one implementation

A person and an agent drive gymrat through the same commands and the same statistical engine.
Every mechanism works the same whether a person drives the loop or the supervisor does. Supervised
mode adds caps and time-left lines; it never changes the verdict a command reaches or the record it
writes. gymrat detects supervised mode from state on disk, never from a flag or an environment
variable, so a manual session cannot observe a supervised-mode side effect. No release may make the
manual loop depend on the supervisor.

A feature that helps one party must not cost the other its guarantees. A person keeps output that
reads at a glance and a confirmation before anything irreversible. An agent keeps machine-readable
output, exit codes that mean the same thing in every release, a refusal that says why, and a log
that survives a compacted context.

### 5. The session log is the truth

Session state is whatever an append-only log on disk says, and every reader reads the log from the
start rather than remembering anything between reads. A crash therefore costs the session nothing,
and neither does a compacted agent context. Anything a later reader must act on is a record. State
a live process needs for itself is disposable: losing it can end a run early, and it can never lose
history.

The log is also a complete trace. Every command run inside a session records what it was asked,
what it decided, and how long it took, refusals included, so a session can be reconstructed after
the fact. A record of an action is not a record of progress: a busy agent checking `status` looks
busy in the trace and nowhere else.

## What gymrat is not

1. **Not a profiler.** gymrat judges whether a change moved a metric. Finding where the time goes
   is the caller's job, with the caller's tools.
2. **Not a harness, for benchmarks or for evaluations.** gymrat runs the project's bench command
   and reads the metrics it prints. It does not time code, define benchmarks, grade answers, manage
   a corpus, or call a judge.
3. **Not a metrics history.** gymrat judges candidates against the baseline in front of it. It does
   not track a metric across sessions or over time, because a trend line over noisy runs invites
   the impressions the tool exists to refuse.
4. **Not a sandbox.** Guardrails under supervision are bounded checks with a reason the agent can
   act on. They make mistakes cheap, not impossible.
5. **Not an agent framework.** The supervisor runs one loop under caps. It does not plan or route
   between models.
