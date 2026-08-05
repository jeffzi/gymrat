/** The largest delay `setTimeout` can hold: milliseconds in a signed 32-bit integer. */
const MAX_TIMER_MS = 2_147_483_647;

/**
 * A timeout is expanded to milliseconds before it reaches `setTimeout`, which
 * silently fires immediately once the delay overflows {@link MAX_TIMER_MS} —
 * turning an over-large timeout into no timeout at all.
 */
export const MAX_TIMEOUT_SECONDS = Math.floor(MAX_TIMER_MS / 1000);
