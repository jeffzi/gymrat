import os from "node:os";

/** The signals a shell or supervisor sends to ask for an orderly shutdown. */
const TERMINATION_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;

type TerminationSignal = (typeof TERMINATION_SIGNALS)[number];

/**
 * The cleanups of every run currently being guarded, in install order.
 *
 * Holding them here — rather than in a closure per run — is what keeps a single
 * handler per signal enough: the handler reads this set when the signal lands,
 * so installing and uninstalling never touches the process listener list.
 */
const activeCleanups = new Set<() => void>();

function createHandler(signal: TerminationSignal): () => void {
  const exitCode = 128 + os.constants.signals[signal];

  return () => {
    for (const cleanup of activeCleanups) {
      try {
        cleanup();
      } catch (error) {
        // oxlint-disable-next-line no-console -- last-resort warning during forced exit
        console.warn("termination cleanup failed:", error);
      }
    }
    process.exit(exitCode);
  };
}

/** The one handler this module ever attaches per signal. */
const HANDLERS = new Map<TerminationSignal, () => void>(
  TERMINATION_SIGNALS.map((signal) => [signal, createHandler(signal)]),
);

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
 * The returned function retires `cleanup`. Call it as soon as the work being
 * guarded is over, so a signal arriving later cannot sweep a run that already
 * finished. The handler itself stays attached and keeps exiting with the same
 * code — it does not hold the event loop open, since Node leaves signal handles
 * unreferenced — and it is reused by every later run, so repeated install and
 * uninstall cycles leave the process listener count where they found it.
 */
export function installTerminationCleanup(cleanup: () => void): () => void {
  activeCleanups.add(cleanup);

  for (const [signal, handler] of HANDLERS) {
    if (!process.listeners(signal).includes(handler)) {
      process.on(signal, handler);
    }
  }

  return () => {
    activeCleanups.delete(cleanup);
  };
}
