# CM kernel optimization — flow and artifacts

## 1. Pipeline

Each phase's output decides whether the next one is worth running. The gates matter more than
the boxes: most of the value is in *not* proceeding.

```
                     ┌──────────────────────────────────────────────┐
   task ────────────►│  0  rig-warden                               │
                     │     competing GPU work? env? noise floor?    │
                     └──────────────────────┬───────────────────────┘
                                            │
                        noise ≥ effect ─────┴──► STOP: "not resolvable on this rig"
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │  1  roofline-analyst                         │
                     │     MEASURED roofs (never spec sheets)       │
                     │     compulsory traffic vs algorithm artifact │
                     │     gap = current / floor                    │
                     └──────────────────────┬───────────────────────┘
                                            │
                ┌───────────────────────────┼───────────────────────────┐
            gap < 1.2x                  1.5 – 3x                    gap > 3x
                │                           │                           │
                ▼                           │                           ▼
     ┌────────────────────────┐             │            ┌────────────────────────┐
     │  2  algorithm-critic   │             │            │  2  algorithm-critic   │
     │     micro-opt is       │             │            │     artifacts dominate │
     │     exhausted          │             │            │     — suspect decomp   │
     └───────────┬────────────┘             │            └───────────┬────────────┘
                 │  redesign / re-policy    │                        │
                 └──────────────────────────┼────────────────────────┘
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │  3  budget-prober                            │
                     │     ablation budget, largest term first      │
                     │     self-check: probe cheaper than removed?  │
                     │     self-check: probe-off == baseline?       │
                     └──────────────────────┬───────────────────────┘
                                            ▼
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  4  optimization loop        (one candidate at a time, biggest term first) ║
    ║                                                                           ║
    ║     bitexact-classifier ──── bit-exact? ──── no ──► ask user               ║
    ║              │                                      (max_diff DIST,       ║
    ║              │ yes                                   not just the max)    ║
    ║              ▼                                                            ║
    ║        implement ──► equivalence-prover ──► measure ──► adopt             ║
    ║                            │                   │                          ║
    ║                    non-vacuous? ─ no ─►        │ no gain ─► REJECT        ║
    ║                    fix the TEST                │                          ║
    ╚════════════════════════════════════╤══════════════════════════════════════╝
                                         ▼
                     ┌──────────────────────────────────────────────┐
                     │  5  range-tuner                              │
                     │     calibrate over the DOMAIN, not a point   │
                     │     guard = relative property, same run      │
                     │     verify guard fails on the bad value      │
                     └──────────────────────┬───────────────────────┘
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │  6  integrator      aboutSHW ──► openvino    │
                     │     diff code-only (kernels DIFFER)          │
                     │     build · real model                       │
                     │     check text AND accepted-token count      │
                     └──────────────────────┬───────────────────────┘
                                            ▼
                          LEDGER ─ write results, including
                          negatives, into BOTH kernels' comments
```

Every arrow that leaves the happy path — STOP, REJECT, "fix the TEST", "ask user" — fired at
least once in the work this was distilled from.

## 2. Where things live and what is versioned

```
  aboutSHW/                                    ← versioned: method + kernel + findings
  ├── .claude/agents/*.md                      the 8 roles
  ├── .claude/skills/cm-kernel-opt/            SKILL.md + this file
  ├── opencl/tests/pageatten/
  │   ├── harness/                             bench_paired · kernel_probe
  │   │   ├── sync.sh                          equiv_template · bench_reduce
  │   │   └── README.md
  │   ├── pa_small_q_ov_exp.cm                 sandbox kernel + LEDGER
  │   └── test_*.py                            correctness + perf guards
  │
  openvino/                                    ← versioned: production
  └── src/plugins/intel_gpu/.../cm/
      ├── pa_small_q.cm                        production kernel + LEDGER
      ├── pa_small_q_finalization.cm
      └── paged_attention{,_gen}.{cpp,hpp}     host policy, rung table
  │
  /home/intel/ceciliapeng/kernel_harness/      ← NOT versioned (sync target, derived)
  /home/intel/bell/.claude/{agents,skills}     ← NOT versioned (symlinks, for discovery)
```

Two rules keep this from rotting:

- **Findings live in the kernel, not in chat.** Both `.cm` files carry a comment ledger with
  measured numbers. That convention predates this workflow and is what let a stale
  "+1 to +3%" verdict be identified as a rig artifact once a stable rig existed. Record
  negatives too, and say whether a change was rejected on *measurement* or on *policy*.
- **The two ledgers must not diverge.** When `integrator` ports a change, it ports the ledger
  entry with it.

## 3. Measurement loop (the part with the container)

```
   edit  aboutSHW/opencl/tests/pageatten/*.cm        (host, versioned)
     │
     ├─► harness/sync.sh [kernel.cm]
     │        └─► /home/intel/ceciliapeng/kernel_harness   (only path the container sees)
     │
     └─► docker exec llm bash -lc 'cd /ceciliapeng/bell/aboutSHW/... && python ...'
              └─► the container has its OWN copy of the kernel tree
                  ⇒ re-sync after EVERY edit or you measure the previous version
```

Bare metal is a fallback only (`LD_LIBRARY_PATH=/home/intel/river`), and its numbers are
provisional: it drifts ~2x within a session.

## 4. Mermaid (for docs/PRs)

```mermaid
flowchart TD
    T[task] --> W[0 rig-warden]
    W -->|noise >= effect| STOP[STOP: not resolvable]
    W --> R[1 roofline-analyst]
    R -->|gap < 1.2x| A[2 algorithm-critic]
    R -->|gap > 3x| A
    R -->|gap 1.5-3x| B[3 budget-prober]
    A -->|redesign / re-policy| B
    B --> L{4 loop: per candidate}
    L --> C[bitexact-classifier]
    C -->|not bit-exact| U[ask user: max_diff distribution]
    U --> I[implement]
    C -->|bit-exact| I
    I --> E[equivalence-prover]
    E -->|vacuous| E2[fix the test] --> E
    E --> M[measure]
    M -->|no gain| REJ[reject + ledger] --> L
    M -->|gain| RT[5 range-tuner]
    RT --> IN[6 integrator: aboutSHW to openvino]
    IN --> LG[ledger in both kernels]
```
