import os from "node:os";

/** The signals a shell or supervisor sends to ask for an orderly shutdown. */
const TERMINATION_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;

/**
 * Run `cleanup` when the process is asked to terminate, then exit.
 *
 * Node's default handling of these signals ends the process outright, so whatever
 * a run left on disk — temporary worktrees here — would survive a Ctrl-C, a closed
 * terminal, or a dropped SSH session. With a handler installed, `cleanup` gets its
 * turn first and the process then exits with the conventional `128 + signum`
 * (130 for SIGINT, 143 for SIGTERM, 129 for SIGHUP), so a shell still reads the
 * run as signal-terminated.
 *
 * The returned function uninstalls the handlers. Call it as soon as the work
 * being guarded is over: a handler left installed would tear down a run that
 * already finished and force-exit an otherwise healthy process. It does not
 * hold the event loop open — Node leaves signal handles unreferenced — so that
 * stale cleanup is the only cost of leaving one attached.
 */
export function installTerminationCleanup(cleanup: () => void): () => void {
  const handlers = TERMINATION_SIGNALS.map((signal) => {
    const exitCode = 128 + os.constants.signals[signal];

    const handler = () => {
      cleanup();
      process.exit(exitCode);
    };

    process.on(signal, handler);

    return { signal, handler, exitCode };
  });

  return () => {
    for (const { signal, handler, exitCode } of handlers) {
      process.removeListener(signal, handler);

      const exitOnly = () => {
        process.exit(exitCode);
      };
      process.on(signal, exitOnly);
    }
  };
}
