# spark-lightgbm

> **About this fork**
> **spark-lightgbm** is a personal fork of [microsoft/SynapseML](https://github.com/microsoft/SynapseML) maintained by Hatosabre, trimmed down to the LightGBM module (`core` + `lightgbm`) with fixes for distributed training. It is not affiliated with Microsoft. The original code is Copyright (c) Microsoft Corporation and the fork's modifications are Copyright (c) 2026 Hatosabre; everything is distributed under the MIT License (see [LICENSE](LICENSE)). Issues and pull requests are not accepted here; please use the upstream project.
>
> spark-lightgbm は microsoft/SynapseML の個人用フォークで、LightGBM モジュールに絞り分散学習の修正を加えたものです。Microsoft とは無関係です。元コードの著作権は Microsoft Corporation、改変部分の著作権は Hatosabre にあり、全体を MIT ライセンスで配布します。Issue / Pull Request は受け付けていません。

Distributed [LightGBM](https://github.com/microsoft/LightGBM) training and inference on Apache Spark, exposed as SparkML estimators (`LightGBMClassifier`, `LightGBMRegressor`, `LightGBMRanker`), with per-row TreeSHAP contributions from the native booster.

## What changed in this fork

Compared with upstream `v1.0.8-spark3.5`:

- **Connection refused fix**: the probed executor port is held open until `LGBM_NetworkInit`, closing the race in which the OS could hand the port to another process. Network retries are raised from 3 to 7 with capped exponential backoff.
- **Driver maintenance mode**: after the network topology is built, the driver keeps answering retried tasks with the cached topology instead of failing them.
- **Coordinator pattern for streaming mode**: one task per executor materializes its partition, pushes rows into a single native dataset and runs training; the MAIN/HELPER task split and the stale `partitionCounts` are gone. Streaming mode always uses barrier execution.
- **Training metrics**: metrics recorded during training are logged every `metricLogPeriod` iterations and readable from the trained model (`getTrainingMetricNames`, `getTrainingMetricTrainValues`, `getTrainingMetricValidValues`).
- **Everything unrelated to LightGBM removed**: `cognitive`, `deep-learning`, `opencv`, `vw`, the website, upstream CI and publishing setup. The SHAP explainers in `core` are kept.

Branches: `v1.0.8-spark3.5` (Spark 3.5 / Scala 2.12, upstream v1.0.8) is the mainline. This branch, `spark3.5`, carries the same fixes on the upstream v1.1.1 line; `spark3.5`, `spark4.0` and `spark4.1` are kept in step.

## Environment

| | `v1.0.8-spark3.5` | `spark3.5` | `spark4.0` / `spark4.1` |
|---|---|---|---|
| Spark | 3.5.0 | 3.5.x | 4.0 / 4.1 |
| Scala | 2.12.17 | 2.12.17 | 2.13.16 / 2.13.17 |
| JDK to build and test | 11 | 11 | 17 (Spark 4 requires 17+) |
| sbt | 1.10.11 | 1.10.11 | 1.10.11 |
| Native LightGBM (`lightgbmlib`) | 3.3.510 | 3.3.510 | 3.3.510 |

Scala 2.12.17 cannot build its compiler bridge on JDK 21, so use JDK 11 (or 17) for the 2.12 branches. The native library needs the OpenMP runtime (`libgomp1` on Debian/Ubuntu) on every executor.

The easiest reproducible environment is the CI image:

```bash
docker build -t synapseml-ci tools/docker/ci            # JDK 11 + sbt 1.10.11 + libgomp1
docker build --build-arg JDK_VERSION=17 -t synapseml-ci tools/docker/ci   # for spark4.x branches
```

## Setup

### Build the JAR

```bash
git clone https://github.com/Hatosabre/spark-lightgbm.git
cd spark-lightgbm && git checkout spark3.5

# fat JAR with lightgbmlib and the core module, Scala/Spark excluded
sbt lightgbm/assembly
# -> lightgbm/target/scala-2.12/synapseml-lightgbm-assembly-<version>.jar

# or inside Docker
docker run --rm -v "$PWD":/workspace synapseml-ci sbt -batch lightgbm/assembly
```

`<version>` is derived from the git tag (`git describe`), for example `1.1.1-spark3.5-<n>-g<sha>`.

### Use it from Spark

Put the assembly JAR on the driver and executor classpath. It must come before any upstream `synapseml` JAR.

```bash
spark-submit --jars synapseml-lightgbm-assembly-<version>.jar my_job.py
# or
spark-shell --jars synapseml-lightgbm-assembly-<version>.jar
```

For PySpark the Python wrappers are thin py4j shims over the JVM classes. Either build them with `sbt lightgbm/packagePython` (needs the conda environment from `environment.yml`) or install the upstream wheel `synapseml==1.0.8` and let the fork's JAR override the JVM side; methods added by this fork are reachable through `model._java_obj`.

### Verify

```bash
sbt lightgbm/Test/compile
sbt getDatasets "lightgbm/testOnly *VerifyCoordinatorAccuracy *VerifyLightGBMClassifierStream"
```

The same two commands run in GitHub Actions (`.github/workflows/ci.yml`) inside the CI image.

## Usage

### Scala

```scala
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMClassifier, LightGBMRegressor}

val classifier = new LightGBMClassifier()
  .setFeaturesCol("features")        // a Vector column, e.g. from VectorAssembler or Featurize
  .setLabelCol("label")
  .setObjective("binary")
  .setNumIterations(200)
  .setLearningRate(0.05)
  .setNumLeaves(63)
  .setDataTransferMode("streaming")  // default; "bulk" is the legacy path
  .setMetricLogPeriod(10)            // log metrics every 10 iterations (default), 0 to disable

val model = classifier.fit(train)
val scored = model.transform(test)   // prediction, rawPrediction, probability

val regressor = new LightGBMRegressor()
  .setObjective("quantile").setAlpha(0.3)
  .setNumIterations(100).setNumLeaves(31)
```

### PySpark

```python
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor

model = LightGBMClassifier(
    featuresCol="features", labelCol="label", objective="binary",
    numIterations=200, learningRate=0.05, numLeaves=63,
    dataTransferMode="streaming",
).fit(train)

scored = model.transform(test)
```

### Validation data and early stopping

Mark validation rows with a boolean column and pass its name; early stopping and the validation metrics use it.

```scala
val df = data.withColumn("isVal", rand() < 0.2)
val m = new LightGBMClassifier()
  .setValidationIndicatorCol("isVal")
  .setEarlyStoppingRound(20)
  .setMetric("auc")
  .fit(df)
```

### Training metrics (added in this fork)

```scala
val names = model.getTrainingMetricNames()                   // e.g. Array("binary_logloss", "auc")
val iters = model.getTrainingMetricIterations()              // iterations at which metrics were recorded
val trainLoss = model.getTrainingMetricTrainValues("binary_logloss")
val validAuc  = model.getTrainingMetricValidValues("auc")    // NaN where no validation data
```

From PySpark: `model._java_obj.getTrainingMetricTrainValues("binary_logloss")`.

### Feature importance and leaf indices

```scala
model.getFeatureImportances("gain")   // or "split"; one value per feature
model.setLeafPredictionCol("leaves").transform(test)   // predicted leaf index per tree
```

### Save and load

Models are SparkML `MLWritable`, and the native booster can be exported on its own.

```scala
model.write.overwrite().save("/path/to/model")
val loaded = LightGBMClassificationModel.load("/path/to/model")
model.saveNativeModel("/path/to/model.txt", overwrite = true)   // plain LightGBM text model
```

### Passing any other LightGBM parameter

Parameters without a setter can be passed as a raw string; they are appended to the native config and take precedence over explicit setters.

```scala
.setPassThroughArgs("force_row_wise=true min_sum_hessian_in_leaf=2e-3")
```

## TreeSHAP

The native booster computes exact TreeSHAP contributions. Set an output column on the trained model and every row gets a dense vector of SHAP values:

```scala
val shapped = model.setFeaturesShapCol("shap").transform(test)
shapped.select("shap").show(1, truncate = false)
```

```python
shapped = model.setFeaturesShapCol("shap").transform(test)
```

Layout of the `shap` vector:

- binary classification and regression: `numFeatures + 1` values; the last element is the expected value (bias). Summing all elements gives the raw margin for the row.
- multiclass: `numClass * (numFeatures + 1)` values, class by class, each block ending with that class's bias.

For a single instance without a DataFrame: `model.getFeatureShaps(features: Vector)` returns the same array; `getDenseFeatureShaps` / `getSparseFeatureShaps` are the PySpark-friendly variants.

Model-agnostic KernelSHAP (`TabularSHAP`, `VectorSHAP`, `TextSHAP` in `com.microsoft.azure.synapse.ml.explainers`) is still available in `core` for comparing against other models, but for LightGBM the native TreeSHAP above is exact and far cheaper.

## Hyperparameters

Setters follow the parameter name (`setNumLeaves`, PySpark keyword `numLeaves`). Defaults are the fork's current values.

### Tree structure and boosting

| Parameter | Default | Meaning |
|---|---|---|
| `objective` | `regression` | `binary`, `multiclass`, `regression`, `quantile`, `huber`, `tweedie`, `lambdarank`, ... (classifier and ranker set their own) |
| `boostingType` | `gbdt` | `gbdt`, `rf`, `dart`, `goss` |
| `numIterations` | 100 | number of boosting rounds (`num_class * numIterations` trees for multiclass) |
| `learningRate` | 0.1 | shrinkage rate |
| `numLeaves` | 31 | max leaves per tree; the main capacity knob |
| `maxDepth` | -1 | max tree depth, -1 = unlimited |
| `minDataInLeaf` | 20 | min rows per leaf; raise to fight overfitting |
| `minSumHessianInLeaf` | 1e-3 | min hessian sum per leaf |
| `minGainToSplit` | 0 | min gain to make a split |
| `maxBin` | 255 | histogram bins per feature |
| `maxDeltaStep` | 0 | caps leaf output; useful for `poisson` / unbalanced targets |

### Regularization and sampling

| Parameter | Default | Meaning |
|---|---|---|
| `lambdaL1`, `lambdaL2` | 0, 0 | L1 / L2 regularization on leaf values |
| `featureFraction` | 1.0 | fraction of features sampled per tree |
| `featureFractionByNode` | 1.0 | fraction of features sampled per node |
| `baggingFraction`, `baggingFreq` | 1.0, 0 | row subsampling and how often to resample (`baggingFreq > 0` enables it) |
| `posBaggingFraction`, `negBaggingFraction` | 1.0, 1.0 | class-wise bagging for binary tasks |
| `dropRate`, `maxDrop`, `skipDrop`, `uniformDrop`, `xGBoostDartMode` | | DART-only controls |
| `topRate`, `otherRate` | 0.2, 0.1 | GOSS-only retain ratios |
| `seed` | | master seed; `baggingSeed`, `featureFractionSeed`, `dataRandomSeed`, `objectiveSeed`, `dropSeed`, `extraSeed` can be set individually |
| `deterministic` | false | reproducible histograms on CPU at some speed cost |

### Task-specific

| Parameter | Estimator | Meaning |
|---|---|---|
| `isUnbalance` | classifier | reweight positive class for unbalanced binary data |
| `numClass` / `maxNumClasses` | classifier | multiclass count (inferred from the label when `multiclass`) |
| `alpha` | regressor | quantile for `quantile`, delta for `huber` |
| `tweedieVariancePower` | regressor | power for `tweedie` |
| `groupCol`, `evalAt` | ranker | query group column and NDCG/MAP cut-offs |
| `weightCol`, `initScoreCol` | all | row weights and initial scores |
| `categoricalSlotIndexes`, `categoricalSlotNames`, `catl2`, `minDataPerGroup` | all | categorical features (by index or slot name) and their regularization |
| `monotoneConstraints`, `monotoneConstraintsMethod`, `monotonePenalty` | all | per-feature monotonic constraints |

### Validation, metrics and early stopping

| Parameter | Default | Meaning |
|---|---|---|
| `validationIndicatorCol` | | boolean column marking validation rows |
| `metric` | | `auc`, `binary_logloss`, `multi_logloss`, `l2`, `ndcg`, ... |
| `earlyStoppingRound` | 0 | stop when the metric has not improved for N rounds (0 = off) |
| `improvementTolerance` | 0 | minimum improvement to count as progress |
| `isProvideTrainingMetric` | false | also evaluate metrics on the training set |
| `metricLogPeriod` | 10 | log and record metrics every N iterations, 0 = off (fork) |
| `verbosity` | -1 | <0 fatal, 0 error, 1 info, >1 debug |

### Distributed execution

| Parameter | Default | Meaning |
|---|---|---|
| `dataTransferMode` | `streaming` | `streaming`: one native dataset per executor fed in micro-batches through the Coordinator task (barrier mode, 1 task per executor). `bulk`: legacy, copies the whole partition into executor memory first |
| `microBatchSize` | 100 | rows per push in streaming mode; lower it for very wide data |
| `maxStreamingOMPThreads` | 16 | OpenMP threads for the native push; the fork raises it to at least the executor's core count to avoid buffer overlap |
| `numTasks` | 0 | number of Spark tasks for training; 0 = derived from the cluster (streaming: executors, bulk: executors x cores) |
| `numThreads` | 0 | native threads per executor; 0 = all cores |
| `useBarrierExecutionMode` | false | barrier scheduling for bulk mode (streaming always uses it) |
| `useSingleDatasetMode` | true | bulk mode only: one dataset per executor instead of per partition |
| `parallelism` | `data_parallel` | `data_parallel` or `voting_parallel` (`topK` applies to voting) |
| `numBatches` | 0 | split training into N sequential batches to reduce peak memory |
| `timeout` | 1200 | seconds to wait for the network ring |
| `defaultListenPort`, `driverListenPort` | 12400, 0 | executor and driver listen ports (0 = random) |
| `binSampleCount`, `samplingMode`, `samplingSubsetSize` | 200000, `subset`, 1000000 | how many rows are sampled to define histogram bins in streaming mode |
| `repartitionByGroupingColumn` | true | ranker: keep each group in one partition |

Operational notes:

- Turn off dynamic allocation and autoscaling for the duration of `fit`; the native ring assumes a fixed set of executors.
- Keep the number of partitions of the training DataFrame at or below `numTasks`; the estimator repartitions when necessary.
- Every executor must be able to open TCP connections to every other executor on `defaultListenPort` and following ports, and to the driver on `driverListenPort`.

## Layout

```
core/       shared SparkML utilities, featurization, explainers (KernelSHAP)
lightgbm/   LightGBM estimators, models, NetworkManager, partition tasks
project/    sbt build definition and Python/R wrapper code generation
docs/       LightGBM, explainer and tuning notebooks kept from upstream
tools/docker/   CI image (tools/docker/ci) and demo images
```
