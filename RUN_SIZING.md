# Sizing a run: how many environments, how long, how much

On a rented box these are one question. This is the arithmetic and the
measured numbers behind it, so a run can be costed before it is started
rather than watched and hoped over.

## The units, which are easy to get wrong

skrl's progress bar counts **timesteps**, and one timestep advances *every*
environment once. So:

```
samples          = timesteps x num_envs
wall clock (s)   = timesteps / (it/s)
policy updates   = timesteps / rollouts          (rollouts = 24 here)
```

Two consequences worth stating, because they pull in opposite directions:

* Doubling `num_envs` doubles the samples per timestep. It does **not** halve
  the timesteps needed — it improves each gradient estimate and widens the
  domain-randomisation coverage per update.
* Doubling `num_envs` does not halve `it/s` either, until the GPU saturates.
  Below the knee it is close to free; above it, it is a straight loss.

So the useful figure of merit is **env-steps/s**, and the right `num_envs` is
the one where that flattens. `scripts/bench_throughput.py` measures it.

## Measured, RTX 6000 Ada 48 GB

Baseline from the first hover run, `Isaac-Drone-Hover-v0`:

| | |
|---|---|
| num_envs | 4096 |
| rate | ~17.5 it/s |
| env-steps/s | ~71,700 |
| GPU memory | ~7.6 GB of 45 GB usable |
| GPU utilisation | 62-83% |

Two standing taxes on this box, both permanent:

* The idle Kit streaming app holds **~3.5 GB of GPU** and ~300% of one CPU.
  It is PID 1's child and killing it takes the container down, so subtract it
  from the budget rather than trying to reclaim it.
* Usable GPU memory is therefore ~45 GB, not 49.

## Worked example

A full default racing run is `timesteps: 50000` in `skrl_cfg.yaml`:

```
at 4096 envs, 17.5 it/s:
    wall clock = 50000 / 17.5      = 2857 s   = 48 min
    samples    = 50000 x 4096      = 205 M
    updates    = 50000 / 24        = 2083
    cost       = 0.8 h x $0.87/h   = $0.70
```

That is the number that reframes the whole budget: a *complete* racing run is
under an hour and under a dollar. The constraint on this project is not
compute, and has not been for a while — it was the plant being wrong, which is
a far more expensive thing to get wrong because it does not announce itself.

## Picking the number

1. Run `scripts/bench_throughput.py --headless` on the box, with nothing else
   using the card. It prints the knee.
2. Use the knee for `--num_envs` unless memory is wanted elsewhere.
3. Compute wall clock from `timesteps / (it/s at that size)` and cost from the
   hourly rate. Do this *before* starting, and write it down — a run whose
   finish time was never estimated is a run nobody notices has hung.

## Hover is deliberately short

A converged station keeper has learned to damp all motion, which is precisely
the opposite of what a racing policy needs. It is a seed, not a product: the
policy weights transfer, the value head does not. Stop it early and judge it
by whether it holds position at all, not by squeezing the last of the reward
out of it.
