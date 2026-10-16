"""
Benchmark harness for the MIP Competition 2027 IIS checker.

Pipeline (see run.py):
  1. read     — pair each solution JSON in --solution-dir with its instance
  2. check    — run checker.verify_subsystem() on the pair, with a chosen backend
  3. report   — collect a results table + <out-dir>/summary.json

The harness only reads solutions; producing them is a separate concern. That
split is deliberate: a benchmark run then measures the checker rather than the
producer.
"""
