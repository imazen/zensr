export const WORKER_WASM_MIB = 64;
// WASM maximum plus bounded transferred tile buffers. JS/driver overhead is separate.
export const WORKER_RESERVATION_MIB = 80;
export function defaultCpuBudget(deviceMemory) {
  return Number.isFinite(deviceMemory) && deviceMemory > 0
    ? Math.max(128, Math.min(2048, deviceMemory * 256))
    : 512;
}
export function planWorkers({ logical, requested = 0, budgetMiB, tiles }) {
  if (
    !Number.isSafeInteger(logical) ||
    logical < 1 ||
    !Number.isSafeInteger(requested) ||
    requested < 0 ||
    !Number.isFinite(budgetMiB) ||
    budgetMiB < 80 ||
    budgetMiB > 2048 ||
    !Number.isSafeInteger(tiles) ||
    tiles < 1
  )
    throw new Error("Invalid CPU worker or memory budget.");
  const count = Math.min(
    logical,
    requested || logical,
    Math.floor(budgetMiB / WORKER_RESERVATION_MIB),
    tiles,
  );
  return {
    count,
    reservedMiB: count * WORKER_RESERVATION_MIB,
    wasmLimitMiB: WORKER_WASM_MIB,
  };
}
