// Copyright (C) Microsoft Corporation. All rights reserved.
// Copyright (C) 2026 Hatosabre. Modifications licensed under the MIT License.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.dataset.{LightGBMDataset, ReferenceDatasetUtils}
import com.microsoft.azure.synapse.ml.lightgbm.swig._
import com.microsoft.ml.lightgbm._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType

import java.util.concurrent.{Callable, ExecutorService, Executors, Future}
import scala.language.existentials

/**
  * Coordinator partition task for streaming mode.
  * Runs 1 task per executor with an internal thread pool for parallel data push.
  * Eliminates the MAIN/HELPER pattern and SharedState synchronization.
  */
class CoordinatorPartitionTask extends BasePartitionTask {

  @transient private var trainDataset: Option[LightGBMDataset] = None
  @transient private var validDataset: Option[LightGBMDataset] = None

  override protected def initializeInternal(ctx: TrainingContext,
                                            shouldExecuteTraining: Boolean,
                                            isEmptyPartition: Boolean): Unit = {
    // Link validation dataset worker so that shouldCalcValidationDataset is set correctly
    if (ctx.hasValidationData && !isEmptyPartition) {
      ctx.sharedState().linkValidationDatasetWorker()
    }
  }

  override protected def preparePartitionDataInternal(ctx: PartitionTaskContext,
                                                      inputRows: Iterator[Row]): PartitionDataState = {
    // Step 1: Determine matrix type (sparse vs dense) by sampling first rows
    val rowIterator = determineMatrixType(ctx, inputRows)

    // Step 2: Materialize the iterator to fix the row count
    val rowBuffer: Array[Row] = rowIterator.toArray
    val totalRows = rowBuffer.length
    log.info(s"Coordinator on partition ${ctx.partitionId}: materialized $totalRows rows")

    if (totalRows == 0) {
      log.warn(s"Coordinator on partition ${ctx.partitionId}: no rows to process")
      return PartitionDataState(None, None)
    }

    // Step 3: Create dataset from serialized reference with exact row count
    val trainingCtx = ctx.trainingCtx
    // Use single-threaded push: eliminates potential thread-safety issues in LightGBM's
    // C++ push API (push_buffers_ overlap, concurrent bin insertion). The push phase is
    // memory-bandwidth-bound so parallel push provides minimal throughput benefit.
    val numPushThreads = 1
    // Ensure maxStreamingOMPThreads >= availableProcessors to prevent push_buffers_ overlap.
    // In LightGBM C++, internal_tid = omp_get_thread_num() + (max_omp_threads * tid).
    // If OMP_NUM_THREADS > max_omp_threads, omp_get_thread_num() can exceed the allocated
    // buffer range, causing data corruption that manifests as SIGSEGV during training.
    val availableProcessors = Runtime.getRuntime.availableProcessors
    val configMaxStreamingOMP = ctx.trainingParams.executionParams.maxStreamingOMPThreads
    val safeMaxStreamingOMPThreads = Math.max(configMaxStreamingOMP, availableProcessors)

    log.info(s"Coordinator on partition ${ctx.partitionId}: creating dataset with " +
      s"totalRows=$totalRows, numPushThreads=$numPushThreads, " +
      s"numCols=${trainingCtx.numCols}, numClass=${ctx.trainingParams.getNumClass}, " +
      s"maxStreamingOMPThreads=$safeMaxStreamingOMPThreads (config=$configMaxStreamingOMP), " +
      s"availableProcessors=$availableProcessors, " +
      s"numTasksPerExecutor=${trainingCtx.numTasksPerExecutor}")

    val dataset = ReferenceDatasetUtils.getInitializedReferenceDatasetForCoordinator(
      trainingCtx.serializedReferenceDataset.get,
      totalRows,
      trainingCtx.datasetParams,
      trainingCtx.hasWeightsAsInt,
      trainingCtx.hasInitialScoresAsInt,
      trainingCtx.hasGroupsAsInt,
      ctx.trainingParams.getNumClass,
      numPushThreads,
      safeMaxStreamingOMPThreads,
      trainingCtx.featureNames,
      trainingCtx.numCols)

    // Step 4: Push rows in parallel using thread pool
    parallelPushRows(ctx, dataset, rowBuffer, numPushThreads)

    trainDataset = Some(dataset)

    // Step 5: Create validation dataset if needed
    if (ctx.shouldCalcValidationDataset) {
      validDataset = Some(generateValidationDataset(ctx, dataset))
    }

    PartitionDataState(None, None)
  }

  override protected def getTrainingDatasetInternal(ctx: PartitionTaskContext,
                                                    dataState: PartitionDataState): LightGBMDataset = {
    val dataset = trainDataset.get
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetMarkFinished(dataset.datasetPtr), "Dataset mark finished")
    dataset
  }

  override protected def getValidationDatasetInternal(ctx: PartitionTaskContext,
                                                      dataState: PartitionDataState,
                                                      referenceDataset: LightGBMDataset): LightGBMDataset = {
    validDataset.get
  }

  private def parallelPushRows(ctx: PartitionTaskContext,
                                dataset: LightGBMDataset,
                                rowBuffer: Array[Row],
                                numThreads: Int): Unit = {
    val totalRows = rowBuffer.length
    val chunkSize = (totalRows + numThreads - 1) / numThreads
    val isSparse = ctx.sharedState.isSparse.get
    val trainingCtx = ctx.trainingCtx

    log.info(s"Coordinator pushing $totalRows rows with $numThreads threads, " +
      s"chunkSize=$chunkSize, sparse=$isSparse")

    if (numThreads <= 1) {
      // Single-threaded push - no thread pool needed
      pushRowChunk(ctx, dataset, rowBuffer, 0, totalRows, 0)
      return
    }

    val pool: ExecutorService = Executors.newFixedThreadPool(numThreads)
    try {
      val futures: Array[Future[Unit]] = (0 until numThreads).map { tid =>
        pool.submit(new Callable[Unit] {
          override def call(): Unit = {
            val startIdx = tid * chunkSize
            val endIdx = Math.min(startIdx + chunkSize, totalRows)
            if (startIdx < totalRows) {
              pushRowChunk(ctx, dataset, rowBuffer, startIdx, endIdx, tid)
            }
          }
        })
      }.toArray

      // Wait for all threads and collect errors
      val errors = futures.flatMap { f =>
        try { f.get(); None }
        catch { case e: java.util.concurrent.ExecutionException => Some(e.getCause) }
      }

      if (errors.nonEmpty) {
        errors.tail.foreach(e => log.error("Additional push error", e))
        throw errors.head
      }
    } finally {
      pool.shutdown()
    }
  }

  private def pushRowChunk(ctx: PartitionTaskContext,
                            dataset: LightGBMDataset,
                            rowBuffer: Array[Row],
                            startIdx: Int,
                            endIdx: Int,
                            threadIndex: Int): Unit = {
    val state = createStreamingState(ctx, dataset, threadIndex)
    try {
      val rows = rowBuffer.slice(startIdx, endIdx).iterator
      if (state.isSparse) {
        pushSparseMicroBatches(state, rows, startIdx, endIdx)
      } else {
        pushDenseMicroBatches(state, rows, startIdx, endIdx)
      }
    } finally {
      state.delete()
    }
  }

  private def createStreamingState(ctx: PartitionTaskContext,
                                    dataset: LightGBMDataset,
                                    threadIndex: Int): StreamingState = {
    StreamingState(ctx, dataset, threadIndex)
  }

  private def pushDenseMicroBatches(state: StreamingState,
                                     inputRows: Iterator[Row],
                                     startIndex: Int,
                                     stopIndex: Int): Unit = {
    var currentStart = startIndex
    while (currentStart < stopIndex) {
      val maxBatchSize = Math.min(state.microBatchSize, stopIndex - currentStart)
      val count =
        if (maxBatchSize == 0) 0
        else loadOneDenseMicroBatchBuffer(state, inputRows, 0, maxBatchSize)
      if (count <= 0) return

      if (state.hasInitialScores && state.microBatchSize != count && state.numInitScoreClasses > 1) {
        (1 until state.numInitScoreClasses).foreach { i =>
          (0 until count).foreach { j =>
            val score = state.initScoreBuffer.getItem(i * state.microBatchSize + j)
            state.initScoreBuffer.setItem(i * count + j, score)
          }
        }
      }
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetPushRowsWithMetadata(
        state.datasetPointer,
        state.featureDataPtr,
        lightgbmlibConstants.C_API_DTYPE_FLOAT64,
        count,
        state.numCols,
        currentStart,
        state.labelPtr,
        state.weightPtr,
        state.initScorePtr,
        state.groupPtr,
        state.threadIndex), "Dataset push dense micro-batch")

      currentStart += count
    }
  }

  private def pushSparseMicroBatches(state: StreamingState,
                                      inputRows: Iterator[Row],
                                      startIndex: Int,
                                      stopIndex: Int): Unit = {
    var currentStart = startIndex
    while (currentStart < stopIndex) {
      val maxBatchSize = Math.min(state.microBatchSize, stopIndex - currentStart)
      val (microBatchRowCount, microBatchElementCount) =
        if (maxBatchSize == 0) (0, 0)
        else loadOneSparseMicroBatchBuffer(state, inputRows, 0, 0, maxBatchSize)
      if (microBatchRowCount <= 0) return

      if (state.hasInitialScores && state.microBatchSize != microBatchRowCount && state.numInitScoreClasses > 1) {
        (1 until state.numInitScoreClasses).foreach { i =>
          (0 until microBatchRowCount).foreach { j =>
            val score = state.initScoreBuffer.getItem(i * state.microBatchSize + j)
            state.initScoreBuffer.setItem(i * microBatchRowCount + j, score)
          }
        }
      }
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetPushRowsByCSRWithMetadata(
        state.datasetPointer,
        state.indptrPtr,
        lightgbmlibConstants.C_API_DTYPE_INT64,
        state.indicesPtr,
        state.valPtr,
        lightgbmlibConstants.C_API_DTYPE_FLOAT64,
        microBatchRowCount + 1,
        microBatchElementCount,
        currentStart,
        state.labelPtr,
        state.weightPtr,
        state.initScorePtr,
        state.groupPtr,
        state.threadIndex), "Dataset push CSR micro-batch")

      currentStart += microBatchRowCount
    }
  }

  private def loadOneDenseMicroBatchBuffer(state: StreamingState,
                                            inputRows: Iterator[Row],
                                            currentCount: Int,
                                            maxBatchCount: Int): Int = {
    var count = currentCount
    while (inputRows.hasNext && count < maxBatchCount) {
      val row = inputRows.next()
      row.getAs[Any](state.featureIndex) match {
        case dense: DenseVector => dense.values.zipWithIndex.foreach { case (x, i) =>
          state.featureDataBuffer.setItem(count * state.numCols + i, x) }
        case sparse: SparseVector => sparse.toArray.zipWithIndex.foreach { case (x, i) =>
          state.featureDataBuffer.setItem(count * state.numCols + i, x) }
      }
      loadOneMetadataRow(state, row, count)
      count += 1
    }
    count
  }

  private def loadOneSparseMicroBatchBuffer(state: StreamingState,
                                             inputRows: Iterator[Row],
                                             batchRowCount: Int,
                                             elementCount: Int,
                                             maxBatchCount: Int): (Int, Int) = {
    var rowCount = batchRowCount
    var elemCount = elementCount
    while (inputRows.hasNext && rowCount < maxBatchCount) {
      val row = inputRows.next()
      val sparseVector = row.getAs[Any](state.featureIndex) match {
        case dense: DenseVector => dense.toSparse
        case sparse: SparseVector => sparse
        case other => throw new Exception(s"Unexpected feature type: ${other.getClass}")
      }

      val rowElementCount = sparseVector.values.length
      sparseVector.values.zipWithIndex.foreach { case (value, i) =>
        state.valBuffer.setItem(elemCount + i, value) }
      sparseVector.indices.zipWithIndex.foreach { case (index, i) =>
        state.indicesBuffer.setItem(elemCount + i, index) }
      elemCount += rowElementCount
      state.indptrBuffer.setItem(rowCount + 1, elemCount)

      loadOneMetadataRow(state, row, rowCount)
      rowCount += 1
    }
    (rowCount, elemCount)
  }

  private def loadOneMetadataRow(state: StreamingState, row: Row, index: Int): Unit = {
    state.labelBuffer.setItem(index, row.getDouble(state.labelIndex).toFloat)
    if (state.hasWeights) state.weightBuffer.setItem(index, row.getDouble(state.weightIndex).toFloat)
    if (state.hasGroups) {
      val groupIdManager = state.ctx.sharedState.groupIdManager
      state.groupBuffer.setItem(index, groupIdManager.getUniqueIdForGroup(row.getAs[Any](state.groupIndex)))
    }
    if (state.hasInitialScores) {
      if (row.schema(state.initScoreIndex).dataType == VectorType)
        row.getAs[DenseVector](state.initScoreIndex).values.zipWithIndex.foreach {
          case (value, i) => state.initScoreBuffer.setItem(index + state.microBatchSize * i, value) }
      else
        state.initScoreBuffer.setItem(index, row.getDouble(state.initScoreIndex))
    }
  }

  private def generateValidationDataset(ctx: PartitionTaskContext,
                                         trainDataset: LightGBMDataset): LightGBMDataset = {
    val validationData = ctx.trainingCtx.validationData.get.value

    val pointer = lightgbmlib.voidpp_handle()
    LightGBMUtils.validate(
      lightgbmlib.LGBM_DatasetCreateByReference(trainDataset.datasetPtr, validationData.length, pointer),
      "Dataset create from reference")

    val datasetPtr = lightgbmlib.voidpp_value(pointer)
    LightGBMUtils.validate(
      lightgbmlib.LGBM_DatasetSetWaitForManualFinish(datasetPtr, 1),
      "Dataset LGBM_DatasetSetWaitForManualFinish")

    lightgbmlib.delete_voidpp(pointer)
    val validDataset = new LightGBMDataset(datasetPtr)
    validDataset.setFeatureNames(ctx.trainingCtx.featureNames, ctx.trainingCtx.numCols)

    // Push validation data (single-threaded, typically small)
    val validState = StreamingState(ctx, validDataset, 0)
    try {
      if (ctx.sharedState.isSparse.get)
        pushSparseMicroBatches(validState, validationData.iterator, 0, validationData.length)
      else
        pushDenseMicroBatches(validState, validationData.iterator, 0, validationData.length)
    } finally {
      validState.delete()
    }

    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetMarkFinished(validDataset.datasetPtr),
      "Dataset mark finished")
    validDataset
  }
}
