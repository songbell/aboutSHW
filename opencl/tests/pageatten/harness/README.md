# CM kernel measurement harness

Support scripts for the `cm-kernel-opt` workflow (`.claude/skills/cm-kernel-opt`, agents in
`.claude/agents/`). Everything here exists because a specific mistake was made without it.

## Environment

Measure **inside the `llm` container**. Bare metal on this box drifts ~2× within a session —
the same config measured 0.52 ms early and 1.59 ms late, which silently flipped the sign of
several A/B tests. The container holds ~0.6%.

The container mounts only `/home/intel/ceciliapeng`, so:

```bash
./sync.sh                          # harness -> /home/intel/ceciliapeng/kernel_harness
./sync.sh ../pa_small_q_ov_exp.cm  # ...plus a kernel snapshot to measure
```

Re-run `sync.sh` after **every** kernel edit. Its container tree
(`/ceciliapeng/bell/aboutSHW/...`) is a separate copy from the one you edit.

Bare-metal fallback needs `LD_LIBRARY_PATH=/home/intel/river` for `libclangFEWrapper.so`.

## Scripts

| script | purpose |
|---|---|
| `bench_paired.py` | Interleaved, min-of-N timing. Alternates configs round by round so the second one is not charged for the first one's heat, and reports the spread so you can see whether an effect is resolvable. |
| `kernel_probe.py` | Compiles the **plugin's** kernel standalone with JIT overrides, injects scalars, and A/Bs a `-D` macro. This is how a plugin-only code path (`HAS_QQ_BIAS`) was measured without running the model — it turned out to be 17% of the kernel and absent from the sandbox copy entirely. |
| `equiv_template.py` | Bit-level A/B equivalence between two kernel variants. Template for the non-negotiable parts: shared inputs, full-buffer `array_equal`, ragged shapes, and a **non-vacuity proof**. |
| `bench_reduce.py` | Sweeps a constant (partition size, reduce split) across shapes — the input to `range-tuner`. |

## Three traps these encode

1. **Fresh random inputs per call.** `_build_small_q_inputs` re-randomises on every
   invocation. Calling it once per side of an A/B produced 0/96 mismatches that looked like a
   kernel bug and was the harness. `equiv_template.py` caches by shape.

2. **A test input that never exercises the change.** An all-ones `qq_bias` masks nothing, so
   a "bit-identical" check under it cannot detect a wrong row↔`t` mapping. Worse, the probe
   chosen to prove the mask *did* fire (`lower_tri`) is exactly what the causal mask already
   applies — so it never could. Always print which inputs actually change the result.

3. **An ablation that costs more than what it removes.** Replacing the DPAS with vector adds
   made the kernel 4× slower; it measured the cost of *not* using DPAS. A probe must be
   validated before its number is used.

## Regression guards

Perf guards live with the tests, not here — e.g.
`test_15k_perf_comparison_ov_exp.py::test_small_q_partition_choice`. They assert **relative**
properties measured in the same run (absolute ms are not portable) and ship with a force-
override env var so the guard itself can be verified in both directions:

```bash
RUN_PA_PERF=1 pytest -s -k partition_choice                       # passes
RUN_PA_PERF=1 PA_FORCE_PARTITION=640 pytest -s -k partition_choice # must fail
```

A guard only ever seen to pass is not known to work.
