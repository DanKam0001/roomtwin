// Web Worker wrapper so long route searches never block the UI.
import { planRouteWithAlternatives, planBuildOrder } from './plan.js';

self.onmessage = (e) => {
  const { id, type, world, items, itemId, entryKey, opts } = e.data;
  try {
    if (type === 'route') {
      const item = items.find((i) => i.id === itemId);
      const others = items.filter((i) => i.id !== itemId);
      self.postMessage({ id, ok: true, result: planRouteWithAlternatives(world, item, others, entryKey, opts) });
    } else if (type === 'order') {
      const result = planBuildOrder(world, items, entryKey, opts, (p) => self.postMessage({ id, progress: p }));
      self.postMessage({ id, ok: true, result });
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.stack || err) });
  }
};
