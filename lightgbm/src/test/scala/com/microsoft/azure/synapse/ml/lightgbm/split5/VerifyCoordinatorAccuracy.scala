// Copyright (C) Microsoft Corporation. All rights reserved.
// Copyright (C) 2026 Hatosabre. Modifications licensed under the MIT License.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm.split5

import com.microsoft.azure.synapse.ml.core.test.base.TestBase
import com.microsoft.azure.synapse.ml.core.test.benchmarks.DatasetUtils
import com.microsoft.azure.synapse.ml.featurize.Featurize
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame

// scalastyle:off magic.number

class VerifyCoordinatorAccuracy extends TestBase {

  val featuresCol = "features"
  val labelCol = "label"
  val numPartitions = 2

  def loadAndFeaturize(file: java.io.File, originalLabelCol: String): DataFrame = {
    val raw = spark.read.format("com.databricks.spark.csv")
      .option("header", "true").option("inferSchema", "true")
      .option("treatEmptyValuesAsNulls", "false")
      .option("delimiter", if (file.getName.endsWith(".tsv")) "\t" else ",")
      .load(file.toString)
      .withColumnRenamed(originalLabelCol, labelCol)

    val featurizer = new Featurize()
      .setOutputCol(featuresCol)
      .setInputCols(raw.columns.filter(_ != labelCol))
      .setOneHotEncodeCategoricals(true)
      .setNumFeatures(raw.columns.length)
    featurizer.fit(raw).transform(raw).select(labelCol, featuresCol)
  }

  def trainAndEvaluate(df: DataFrame, mode: String, objective: String): (Double, Double) = {
    val Array(train, test) = df.randomSplit(Array(0.8, 0.2), seed = 42)
    val model = new LightGBMClassifier()
      .setFeaturesCol(featuresCol)
      .setLabelCol(labelCol)
      .setObjective(objective)
      .setDataTransferMode(mode)
      .setNumIterations(100)
      .setNumLeaves(31)
      .setVerbosity(-1)
      .fit(train)

    val predictions = model.transform(test)
    val accuracy = new MulticlassClassificationEvaluator()
      .setLabelCol(labelCol).setPredictionCol("prediction").setMetricName("accuracy")
      .evaluate(predictions)
    val auc = new BinaryClassificationEvaluator()
      .setLabelCol(labelCol).setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
      .evaluate(predictions)
    (accuracy, auc)
  }

  test("Compare Coordinator (streaming) vs Bulk accuracy on PimaIndian") {
    val df = loadAndFeaturize(DatasetUtils.binaryTrainFile("PimaIndian.csv"), "Diabetes mellitus")
      .cache()
    df.count() // materialize cache

    val (streamAcc, streamAuc) = trainAndEvaluate(df, "streaming", "binary")
    val (bulkAcc, bulkAuc)     = trainAndEvaluate(df, "bulk", "binary")

    println("=" * 60)
    println("PimaIndian (binary classification)")
    println(f"  Streaming (Coordinator): accuracy=$streamAcc%.4f  AUC=$streamAuc%.4f")
    println(f"  Bulk:                    accuracy=$bulkAcc%.4f  AUC=$bulkAuc%.4f")
    println(f"  Difference:              accuracy=${streamAcc - bulkAcc}%.4f  AUC=${streamAuc - bulkAuc}%.4f")
    println("=" * 60)

    assert(streamAcc > 0.7, s"Streaming accuracy too low: $streamAcc")
    assert(streamAuc > 0.7, s"Streaming AUC too low: $streamAuc")
    assert(Math.abs(streamAcc - bulkAcc) < 0.05, s"Accuracy gap too large: ${streamAcc - bulkAcc}")
  }

  test("Compare Coordinator (streaming) vs Bulk accuracy on banknote") {
    val df = loadAndFeaturize(DatasetUtils.binaryTrainFile("data_banknote_authentication.csv"), "class")
      .cache()
    df.count()

    val (streamAcc, streamAuc) = trainAndEvaluate(df, "streaming", "binary")
    val (bulkAcc, bulkAuc)     = trainAndEvaluate(df, "bulk", "binary")

    println("=" * 60)
    println("Banknote (binary classification)")
    println(f"  Streaming (Coordinator): accuracy=$streamAcc%.4f  AUC=$streamAuc%.4f")
    println(f"  Bulk:                    accuracy=$bulkAcc%.4f  AUC=$bulkAuc%.4f")
    println(f"  Difference:              accuracy=${streamAcc - bulkAcc}%.4f  AUC=${streamAuc - bulkAuc}%.4f")
    println("=" * 60)

    assert(streamAcc > 0.9, s"Streaming accuracy too low: $streamAcc")
    assert(streamAuc > 0.9, s"Streaming AUC too low: $streamAuc")
  }

  test("Compare Coordinator (streaming) vs Bulk accuracy on breastCancer") {
    val df = loadAndFeaturize(DatasetUtils.binaryTrainFile("breast-cancer.train.csv"), "Label")
      .cache()
    df.count()

    val (streamAcc, streamAuc) = trainAndEvaluate(df, "streaming", "binary")
    val (bulkAcc, bulkAuc)     = trainAndEvaluate(df, "bulk", "binary")

    println("=" * 60)
    println("BreastCancer (binary classification)")
    println(f"  Streaming (Coordinator): accuracy=$streamAcc%.4f  AUC=$streamAuc%.4f")
    println(f"  Bulk:                    accuracy=$bulkAcc%.4f  AUC=$bulkAuc%.4f")
    println(f"  Difference:              accuracy=${streamAcc - bulkAcc}%.4f  AUC=${streamAuc - bulkAuc}%.4f")
    println("=" * 60)

    assert(streamAcc > 0.9, s"Streaming accuracy too low: $streamAcc")
    assert(streamAuc > 0.9, s"Streaming AUC too low: $streamAuc")
  }
}
