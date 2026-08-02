/**
 * Returns the middle element for an odd-length array, or the average of the
 * two middle elements for an even-length array. Does not mutate `values`.
 *
 * @throws {Error} If `values` is empty.
 */
export function computeMedian(values: readonly number[]): number {
  /* v8 ignore if -- defensive check; never called with empty array */
  if (values.length === 0) {
    throw new Error("Cannot compute median of empty array");
  }

  const sorted = values.toSorted((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

/** Half the range of `values`: `(max - min) / 2`. */
export function computeHalfRange(values: readonly number[]): number {
  return (Math.max(...values) - Math.min(...values)) / 2;
}
