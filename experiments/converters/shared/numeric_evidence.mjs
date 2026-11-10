/** Serialize finite native binary64 values without losing the sign of zero. */
export function numeric(values) {
  const visit = value => {
    if (Array.isArray(value)) return value.map(visit);
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new TypeError('Finite native numeric scalar required');
    }
    return value;
  };
  return { values: visit(values), dtype: 'binary64' };
}

export function stringifyEvidence(value) {
  if (typeof JSON.rawJSON !== 'function') {
    throw new Error('Node.js with JSON.rawJSON support is required; the capture uses Node 24');
  }
  return JSON.stringify(value, (_key, item) =>
    typeof item === 'number' && Object.is(item, -0) ? JSON.rawJSON('-0.0') : item,
  2) + '\n';
}
