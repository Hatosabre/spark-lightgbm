// Copyright (C) Microsoft Corporation. All rights reserved.
// Copyright (C) 2026 Hatosabre. Modifications licensed under the MIT License.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.booster.LightGBMBooster
import com.microsoft.azure.synapse.ml.lightgbm.dataset.LightGBMDataset
import com.microsoft.azure.synapse.ml.lightgbm.params.BaseTrainParams
import org.slf4j.Logger

import java.io._
import scala.annotation.tailrec

private object TrainUtils extends Serializable {

  def createBooster(trainParams: BaseTrainParams,
                    trainDataset: LightGBMDataset,
                    validDatasetOpt: Option[LightGBMDataset]): LightGBMBooster = {
    // Create the booster
    val parameters = trainParams.toString()
    val booster = new LightGBMBooster(trainDataset, parameters)
     trainParams.generalParams.modelString.foreach { modelStr =>
      booster.mergeBooster(modelStr)
    }
    validDatasetOpt.foreach { dataset =>
      booster.addValidationDataset(dataset)
    }
    booster
  }

  def beforeTrainIteration(state: PartitionTaskTrainingState, log: Logger): Unit = {
    if (state.ctx.trainingParams.delegate.isDefined) {
      state.ctx.trainingParams.delegate.get.beforeTrainIteration(
        state.ctx.trainingCtx.batchIndex,
        state.ctx.partitionId,
        state.iteration,
        log,
        state.ctx.trainingParams,
        state.booster,
        state.ctx.trainingCtx.hasValidationData)
    }
  }

  def afterTrainIteration(state: PartitionTaskTrainingState,
                          log: Logger,
                          trainEvalResults: Option[Map[String, Double]],
                          validEvalResults: Option[Map[String, Double]]): Unit = {
    val ctx = state.ctx
    val trainingCtx = ctx.trainingCtx
    if (ctx.trainingParams.delegate.isDefined) {
      ctx.trainingParams.delegate.get.afterTrainIteration(
        trainingCtx.batchIndex,
        ctx.partitionId,
        state.iteration,
        log,
        trainingCtx.trainingParams,
        state.booster,
        trainingCtx.hasValidationData,
        state.isFinished,
        trainEvalResults,
        validEvalResults)
    }
  }

  def getLearningRate(state: PartitionTaskTrainingState, log: Logger): Double = {
    state.ctx.trainingParams.delegate match {
      case Some(delegate) => delegate.getLearningRate(state.ctx.trainingCtx.batchIndex,
                                                      state.ctx.partitionId,
                                                      state.iteration,
                                                      log,
                                                      state.ctx.trainingParams,
                                                      state.learningRate)
      case None => state.learningRate
    }
  }

  def updateOneIteration(state: PartitionTaskTrainingState, log: Logger): Unit = {
    log.debug(s"LightGBM task starting iteration ${state.iteration}")
    val fobj = state.ctx.trainingParams.objectiveParams.fobj
    if (fobj.isDefined) {
      val (gradient, hessian) = fobj.get.getGradient(
        state.booster.innerPredict(0, state.ctx.trainingCtx.isClassification),
        state.booster.trainDataset.get)
      state.isFinished = state.booster.updateOneIterationCustom(gradient, hessian)
    } else {
      state.isFinished = state.booster.updateOneIteration()
    }
  }

  def executeTrainingIterations(state: PartitionTaskTrainingState, log: Logger): Option[Int] = {
    @tailrec
    def iterationLoop(maxIterations: Int): Option[Int] = {
      beforeTrainIteration(state, log)
      val newLearningRate = getLearningRate(state, log)
      if (newLearningRate != state.learningRate) {
        log.info(s"LightGBM task calling booster.resetParameter to reset learningRate" +
          s" (newLearningRate: $newLearningRate)")
        state.booster.resetParameter(s"learning_rate=$newLearningRate")
        state.learningRate = newLearningRate
      }

      updateOneIteration(state, log)

      val trainEvalResults: Option[Map[String, Double]] =
        if (state.ctx.trainingCtx.isProvideTrainingMetric && !state.isFinished) getTrainEvalResults(state, log)
        else None

      val validEvalResults: Option[Map[String, Double]] =
        if (state.ctx.trainingCtx.hasValidationData && !state.isFinished) getValidEvalResults(state, log)
        else None

      // Record structured metric history for every iteration
      if (trainEvalResults.isDefined || validEvalResults.isDefined) {
        state.metricHistory += ((state.iteration,
          trainEvalResults.getOrElse(Map.empty),
          validEvalResults.getOrElse(Map.empty)))
      }

      if (shouldLogMetrics(state) && (trainEvalResults.isDefined || validEvalResults.isDefined)) {
        val line = formatIterationMetrics(state.iteration, trainEvalResults, validEvalResults)
        state.metricLog += line
        log.info(line)
      }

      // Append early stopping message and best iteration's scores after the final metrics line
      state.bestIterationResult.foreach { bestIter =>
        val earlyStopMsg = "Early stopping, best iteration is " + bestIter
        state.metricLog += earlyStopMsg
        // Find the best iteration's metrics from history and display them
        state.metricHistory.find(_._1 == bestIter).foreach { case (iter, trainMap, validMap) =>
          val bestLine = formatIterationMetrics(iter,
            if (trainMap.nonEmpty) Some(trainMap) else None,
            if (validMap.nonEmpty) Some(validMap) else None)
          state.metricLog += bestLine
        }
      }

      afterTrainIteration(state, log, trainEvalResults, validEvalResults)

      state.iteration = state.iteration + 1
      if (!state.isFinished && state.iteration < maxIterations) {
        iterationLoop(maxIterations)  // tail recursion
      } else {
        state.bestIterationResult
      }
    }

    log.info(s"Beginning training on LightGBM Booster for task ${state.ctx.taskId}, partition ${state.ctx.partitionId}")
    state.ctx.measures.markTrainingIterationsStart()
    val result = iterationLoop(state.ctx.trainingParams.generalParams.numIterations)
    state.ctx.measures.markTrainingIterationsStop()
    result
  }

  def getTrainEvalResults(state: PartitionTaskTrainingState, log: Logger): Option[Map[String, Double]] = {
    val evalResults: Array[(String, Double)] = state.booster.getEvalResults(state.evalNames, 0)
    Option(Map(evalResults: _*))
  }

  def getValidEvalResults(state: PartitionTaskTrainingState, log: Logger): Option[Map[String, Double]] = {
    val evalResults: Array[(String, Double)] = state.booster.getEvalResults(state.evalNames, 1)
    val results: Array[(String, Double)] = evalResults.zipWithIndex.map { case ((evalName, evalScore), index) =>
      val cmp =
        if (evalName.startsWith("auc")
            || evalName.startsWith("ndcg@")
            || evalName.startsWith("map@")
            || evalName.startsWith("average_precision"))
          (x: Double, y: Double, tol: Double) => x - y > tol
        else
          (x: Double, y: Double, tol: Double) => x - y < tol
      if (state.bestScores(index) == null
          || cmp(evalScore, state.bestScore(index), state.ctx.trainingCtx.improvementTolerance)) {
        state.bestScore(index) = evalScore
        state.bestIteration(index) = state.iteration
        state.bestScores(index) = evalResults.map(_._2)
      }

      (evalName, evalScore)
    }

    // Early stopping: trigger only when ALL metrics have stopped improving
    if (state.evalCounts > 0 &&
        (0 until state.evalCounts).forall(i =>
          state.bestScores(i) != null &&
          state.iteration - state.bestIteration(i) >= state.ctx.trainingCtx.earlyStoppingRound)) {
      state.isFinished = true
      // Use the most recent best iteration across all metrics
      val bestIter = state.bestIteration.max
      log.info("Early stopping, best iteration is " + bestIter)
      state.bestIterationResult = Some(bestIter)
    }

    Option(Map(results: _*))
  }

  private def shouldLogMetrics(state: PartitionTaskTrainingState): Boolean = {
    val period = state.ctx.trainingCtx.metricLogPeriod
    if (period <= 0) false
    else {
      val maxIter = state.ctx.trainingParams.generalParams.numIterations
      state.iteration % period == 0 || state.isFinished || state.iteration >= maxIter - 1
    }
  }

  private def formatIterationMetrics(iteration: Int,
                                    trainResults: Option[Map[String, Double]],
                                    validResults: Option[Map[String, Double]]): String = {
    val sb = new StringBuilder(s"[${iteration}]")
    trainResults.foreach(_.foreach { case (name, value) =>
      sb.append(s"\ttrain's $name: ${"%.6f".format(value)}")
    })
    validResults.foreach(_.foreach { case (name, value) =>
      sb.append(s"\tvalid's $name: ${"%.6f".format(value)}")
    })
    sb.toString()
  }

  def beforeGenerateTrainDataset(ctx: PartitionTaskContext, log: Logger): Unit = {
    val trainingCtx = ctx.trainingCtx
    if (trainingCtx.trainingParams.delegate.isDefined) {
      trainingCtx.trainingParams.delegate.get.beforeGenerateTrainDataset(
        trainingCtx.batchIndex,
        ctx.partitionId,
        trainingCtx.columnParams,
        trainingCtx.schema,
        log,
        trainingCtx.trainingParams)
    }
  }

  def afterGenerateTrainDataset(ctx: PartitionTaskContext, log: Logger): Unit = {
    val trainingCtx = ctx.trainingCtx
    if (trainingCtx.trainingParams.delegate.isDefined) {
      trainingCtx.trainingParams.delegate.get.afterGenerateTrainDataset(
        trainingCtx.batchIndex,
        ctx.partitionId,
        trainingCtx.columnParams,
        trainingCtx.schema,
        log,
        trainingCtx.trainingParams)
    }
  }

  def beforeGenerateValidDataset(ctx: PartitionTaskContext, log: Logger): Unit = {
    val trainingCtx = ctx.trainingCtx
    if (ctx.trainingCtx.trainingParams.delegate.isDefined) {
      trainingCtx.trainingParams.delegate.get.beforeGenerateValidDataset(
        trainingCtx.batchIndex,
        ctx.partitionId,
        trainingCtx.columnParams,
        trainingCtx.schema,
        log,
        trainingCtx.trainingParams)
    }
  }

  def afterGenerateValidDataset(ctx: PartitionTaskContext, log: Logger): Unit = {
    val trainingCtx = ctx.trainingCtx
    if (trainingCtx.trainingParams.delegate.isDefined) {
      trainingCtx.trainingParams.delegate.get.afterGenerateValidDataset(
        trainingCtx.batchIndex,
        ctx.partitionId,
        trainingCtx.columnParams,
        trainingCtx.schema,
        log,
        trainingCtx.trainingParams)
    }
  }
}
