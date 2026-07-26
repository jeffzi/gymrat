import os from "node:os";

/** The signals a shell or supervisor sends to ask for an orderly shutdown. */
const TERMINATION_SIGNALS = ["SIGINT", "SIGTERM"] as const;

/**
 * Run `cleanup` when the process is asked to terminate, then exit.
 *
 * Node's default SIGINT/SIGTERM handling ends the process outright, so whatever
 * a run left on disk — temporary worktrees here — would survive a Ctrl-C. With a
 * handler installed, `cleanup` gets its turn first and the process then exits with
 * the conventional `128 + signum` (130 for SIGINT, 143 for SIGTERM), so a shell
 * still reads the run as signal-terminated.
 *
 * The returned function uninstalls the handlers. Call it as soon as the work
 * being guarded is over: a handler left installed would tear down a run that
 * already finished and force-exit an otherwise healthy process. It does not
 * hold the event loop open — Node leaves signal handles unreferenced — so that
 * stale cleanup is the only cost of leaving one attached.
 */
export function installTerminationCleanup(cleanup: () => void): () => void {
  const uninstallers = TERMINATION_SIGNALS.map((signal) => {
    const handler = () => {
      cleanup();
      process.exit(128 + os.constants.signals[signal]);
    };

    process.on(signal, handler);

    return () => {
      process.removeListener(signal, handler);
    };
  });

  return () => {
    for (const uninstall of uninstallers) {
      uninstall();
    }
  };
}
