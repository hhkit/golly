Polyhedral Analysis of GPU Kernels


Analysis:
- Barrier divergence
- Race detection
  - Repair?
- Barrier elimination
- Bank conflicts

Issues with polyhedral model:
- Cannot represent post-parallelized code

End goal (PhD):
- Code generation from GSCoP
  Meant as an analytical IR for other polyhedral generators 
  - NOT meant for direct CUDA kernel translation

existing GPU polyhedral work
polyhedral -> GPU: PPCG

```cpp

S;
for (int i = N / 2; i > 0; i /= 2) {
    L;
}  
E;

// We observe three kinds of phases:
// S -> L
// L -> L
// L -> E
// We can fake holes for each of these phases programmatically
// Why not just use Z3 again?
```

* pw_aff = piecewise affine

* mem2reg is required for most analyses
* loop-rotate rotates for loops into do-loops, executing the body at least once.

* scalar evolution only works with mem2reg and loop-rotate, allowing us to extract bounds.

Take note that bounds must be pw affine, so it may not be important to use SCEV. However, loop detection is probably required at minimum.

In Polly, a SCoP is always constrained by a loop.
But we want to include single instructions.