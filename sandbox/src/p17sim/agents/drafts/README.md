# Model A draft (pending)

An early, untrained draft of Model A (retrieval-augmented hierarchical planner), written before the
system was finished. It is **not wired into the system** and has no trained artifacts. It runs on the
old in-memory shell interface, not on the workspace runtime.

When Model A is built, it plugs into the `model_a` slot in `aios/generators.py` and is trained on the
export from `p17sim export-training`.
