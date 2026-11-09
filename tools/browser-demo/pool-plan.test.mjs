import { test } from "node:test";
import assert from "node:assert/strict";
import { planWorkers, defaultCpuBudget } from "./pool-plan.mjs";
test("CPU pool follows cores, explicit requests, image work and aggregate reservation", () => {
  assert.equal(
    planWorkers({ logical: 20, budgetMiB: 2048, tiles: 768 }).count,
    20,
  );
  assert.equal(
    planWorkers({ logical: 20, budgetMiB: 512, tiles: 768 }).count,
    6,
  );
  assert.equal(
    planWorkers({ logical: 20, budgetMiB: 2048, tiles: 3 }).count,
    3,
  );
  assert.equal(
    planWorkers({ logical: 20, requested: 4, budgetMiB: 2048, tiles: 768 })
      .reservedMiB,
    320,
  );
  assert.equal(
    planWorkers({ logical: 4, requested: 20, budgetMiB: 2048, tiles: 768 })
      .count,
    4,
  );
  assert.throws(() => planWorkers({ logical: 4, budgetMiB: 79, tiles: 768 }));
  assert.equal(defaultCpuBudget(8), 2048);
  assert.equal(defaultCpuBudget(2), 512);
  assert.equal(defaultCpuBudget(), 512);
});
