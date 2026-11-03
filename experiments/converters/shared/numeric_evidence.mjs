/** Serialize native numeric outputs with explicit precision and IEEE bit patterns. */
export function numeric(values, dtype = 'binary64') {
  const bits = [];
  const visit = value => {
    if (Array.isArray(value)) return value.map(visit);
    if (typeof value !== 'number') throw new TypeError('Numeric scalar required');
    const buffer = Buffer.alloc(dtype === 'binary32' ? 4 : 8);
    if (dtype === 'binary32') buffer.writeFloatBE(value);
    else buffer.writeDoubleBE(value);
    bits.push(buffer.toString('hex'));
    const native = dtype === 'binary32' ? buffer.readFloatBE() : buffer.readDoubleBE();
    return Number.isFinite(native)
      ? native
      : { nonfinite: Number.isNaN(native) ? 'nan' : native > 0 ? '+inf' : '-inf' };
  };
  const shape = value =>
    !Array.isArray(value) ? [] : [value.length, ...(value.length ? shape(value[0]) : [])];
  return { values: visit(values), dtype, shape: shape(values), ieee754_hex_be: bits };
}
