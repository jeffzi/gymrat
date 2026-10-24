# Telemetry reference

Every attribute gymrat sets on the OpenTelemetry spans and span events it exports, live or
through `gymrat export`. The README's
[Monitoring](https://github.com/jeffzi/gymrat/blob/main/README.md#monitoring) section describes
the span structure.

## Attributes

"`<name>` event" below means a span event named after the corresponding session log or
supervisor event log record; see the [event reference](event-reference.md) for those record types.

| Attribute                      | Type   | Appears on                       | Description                                                                                                               |
| ------------------------------ | ------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `gymrat.session.id`            | string | session, run, command            | Unique identifier for the session.                                                                                        |
| `gymrat.session.branch`        | string | session                          | Git branch created for the session.                                                                                       |
| `gymrat.run.head_sha`          | string | run                              | Git HEAD commit hash (SHA) at launch time.                                                                                |
| `gymrat.run.max_minutes`       | float  | run                              | Wall-clock timeout in minutes.                                                                                            |
| `gymrat.run.max_usd`           | float  | run (when set)                   | Spend cap in US dollars, if configured.                                                                                   |
| `gymrat.run.effort`            | string | run (when set)                   | Effort level, if specified.                                                                                               |
| `gymrat.run.cost_usd`          | float  | run (at end)                     | Cumulative cost of the run in US dollars.                                                                                 |
| `gymrat.run.ended_by`          | string | run (at end, live export only)   | How the run ended: `session`, `wall-clock`, `spend-cap`, `guard`, `stop-condition`, or `hook-failure`.                    |
| `gymrat.run.end_reason`        | string | run (when set, live export only) | For `guard`, the guard reason; for a cap, the cap name; for `stop-condition` and `hook-failure`, the condition's summary. |
| `gymrat.run.duration_ms`       | float  | run (at end, live export only)   | Wall-clock milliseconds the run took.                                                                                     |
| `gen_ai.request.model`         | string | run (when set)                   | Model requested for the run, if specified.                                                                                |
| `gen_ai.provider.name`         | string | run                              | Model provider (currently always `anthropic`).                                                                            |
| `gymrat.command.name`          | string | command                          | Name of the CLI command that ran.                                                                                         |
| `gymrat.command.exit_code`     | int    | command                          | Process-style exit code: 0 success, 1 or 2 failure.                                                                       |
| `gymrat.command.reason`        | string | command (when set)               | Why the command exited non-zero, when it did.                                                                             |
| `gymrat.command.duration_ms`   | int    | command                          | Wall-clock milliseconds the command took.                                                                                 |
| `gymrat.command.args.<key>`    | scalar | command (per arg)                | One attribute per argument the command was invoked with.                                                                  |
| `gymrat.iteration.seq`         | int    | command, events                  | Iteration sequence number the record belongs to.                                                                          |
| `gymrat.iteration.outcome`     | string | iteration event                  | Overall iteration outcome: improved, regressed, or no-signal.                                                             |
| `gymrat.iteration.delta_pct`   | float  | iteration event                  | Percentage change from baseline for the primary metric.                                                                   |
| `gymrat.turn.session_cost_usd` | float  | turn_end event                   | Cumulative session cost in US dollars when the turn ended.                                                                |
| `gymrat.turn.origin`           | string | turn_end event                   | Whether the turn was agent-generated or injected by the supervisor.                                                       |
| `gymrat.turn.budget_exhausted` | bool   | turn_end event                   | Whether the turn hit the `--max-usd` spend cap the agent backend enforces.                                                |
| `gymrat.follow_up.action`      | string | follow_up event                  | Supervisor action taken after the turn.                                                                                   |
| `gymrat.follow_up.reason`      | string | follow_up event                  | Reason for the action, if applicable.                                                                                     |
| `gymrat.cap.name`              | string | cap event                        | Which supervision cap fired.                                                                                              |
| `gymrat.keep.status`           | string | keep event                       | Whether the iteration was committed or blocked.                                                                           |
| `gymrat.keep.reason`           | string | keep event (when set)            | Why the keep was blocked, when status is blocked.                                                                         |
| `gymrat.keep.commit`           | string | keep event (when set)            | Git commit SHA when status is committed.                                                                                  |
| `gymrat.keep.message`          | string | keep event (when set)            | Commit message when status is committed.                                                                                  |
| `gymrat.hook.stage`            | string | hook event                       | Whether the hook ran before or after the iteration.                                                                       |
| `gymrat.hook.exit_code`        | int    | hook event                       | Process exit code of the hook command.                                                                                    |
| `gymrat.hook.duration_ms`      | float  | hook event                       | Wall-clock milliseconds the hook command ran.                                                                             |
| `gymrat.hook.stdout_bytes`     | int    | hook event                       | Bytes the hook command wrote to stdout.                                                                                   |
| `gymrat.hook.stderr_bytes`     | int    | hook event (when set)            | Bytes the hook command wrote to stderr.                                                                                   |
| `gymrat.hook.timed_out`        | bool   | hook event                       | Whether the hook command exceeded its timeout.                                                                            |
| `gymrat.baseline.label`        | string | baseline event                   | Human-readable label for the baseline measurement.                                                                        |
| `gymrat.baseline.duration_ms`  | float  | baseline event (when set)        | Wall-clock milliseconds the baseline measurement took.                                                                    |
| `gymrat.finalize.branch`       | string | finalize event                   | Git branch the finalize squash landed on.                                                                                 |
| `gymrat.finalize.commit`       | string | finalize event                   | Git commit SHA of the squash commit.                                                                                      |
| `gymrat.finalize.message`      | string | finalize event                   | Commit message of the squash commit.                                                                                      |
| `gymrat.stop.message`          | string | stop event                       | Reason the stop was requested.                                                                                            |
