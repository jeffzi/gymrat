/**
 * Returns the middle element for an odd-length array, or the average of the
 * two middle elements for an even-length array. Does not mutate `values`.
 *
 * @throws {Error} If `values` is empty.
 */
export function computeMedian(values: readonly number[]): number {
  /* istanbul ignore if -- defensive check; never called with empty array */
  if (values.length === 0) {
    throw new Error("Cannot compute median of empty array");
  }

  const sorted = values.toSorted((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

/**
 * Half the range of `values`: `(max - min) / 2`.
 *
 * A non-finite sample leaves the range undefined, so the result is `NaN` — the
 * value a report prints blank and the geomean excludes by name. Comparisons
 * against `NaN` are always false, so the scan must reject such a sample
 * explicitly: dropping it would understate the spread, and an all-`NaN` run
 * would collapse to `-Infinity`, which reads downstream as a real measurement.
 *
 * @throws {Error} If `values` is empty.
 */
export function computeHalfRange(values: readonly number[]): number {
  /* istanbul ignore if -- defensive check; never called with empty array */
  if (values.length === 0) {
    throw new Error("Cannot compute half-range of empty array");
  }
  // Scanned rather than spread into `Math.max`/`Math.min`: a long enough run of
  // samples exceeds the number of arguments a call accepts, which throws.
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      return Number.NaN;
    }
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  }
  return (max - min) / 2;
}
