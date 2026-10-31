## Summary

<!-- 何を、なぜ変えたか / What changed and why -->

## Related

<!-- 関連 Issue / PR。クローズするなら `Closes #123` -->

## Test plan

- [ ] `sbt Test/compile`
- [ ] `sbt "lightgbm/testOnly *VerifyCoordinatorAccuracy *VerifyLightGBMClassifierStream"` (or the Docker CI job is green)
- [ ] Distributed run, if `NetworkManager` / partition tasks / `LightGBMBase` changed: environment, executors x cores, result

## Checklist

- [ ] Title follows `type: subject` (`feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`, `ci`)
- [ ] Based on `v1.0.8-spark3.5` (or the matching `spark3.5` / `spark4.x` line) and ported to the other lines if needed
- [ ] New or substantially modified files carry the fork copyright line (see `LICENSE`)
