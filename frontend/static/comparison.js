// Highlight independent trade-offs, preserving ties and missing references.
window.PrismRAWComparison = data => [
  {title: 'Highest reconstruction fidelity', key: 'psnr', higher: true, unit: 'dB'},
  {title: 'Lowest colour error', key: 'delta_e_00', higher: false, unit: 'ΔE00'},
  {title: 'Fastest processing', key: 'time', higher: false, unit: 'ms'}
].map(spec => {
  const values = Object.entries(data.processing_ms).flatMap(([method, time]) => {
    let value = spec.key === 'time' ? time : data.metrics?.[method]?.[spec.key];
    if (spec.key === 'psnr' && value === null && data.metrics?.[method]) value = Infinity;
    return typeof value === 'number' && !Number.isNaN(value) ? [{method, value}] : [];
  });
  if (!values.length) return {title: spec.title, methods: [], value: null};
  const best = (spec.higher ? Math.max : Math.min)(...values.map(row => row.value));
  const methods = values.filter(row => row.value === best || Math.abs(row.value - best) < 1e-9).map(row => row.method);
  return {title: spec.title, methods, value: best, detail: `${best === Infinity ? '∞' : best.toFixed(spec.key === 'time' ? 0 : 2)} ${spec.unit}${methods.length > 1 ? ' · tied' : ''}`};
});
