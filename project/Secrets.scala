// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

import spray.json.*
import spray.json.DefaultJsonProtocol.*

import java.io.{File, IOException}
import scala.io.Source
import scala.sys.process.*

//scalastyle:off field.name
object Secrets {
  private val KvName = "mmlspark-keys"
  private val SubscriptionID = "e342c2c0-f844-4b18-9208-52c8c234c30e"
  private val EnablePublishEnvVar = "SYNAPSEML_ENABLE_PUBLISH"

  lazy private val publishingEnabled: Boolean = sys.env.getOrElse(EnablePublishEnvVar, "false").toBoolean

  protected def exec(command: String, maxRetries: Int = 2, attempt: Int = 0): String = {
    val osCommand = sys.props("os.name").toLowerCase match {
      case x if x contains "windows" => Seq("cmd", "/C") ++ Seq(command)
      case _ => Seq("bash", "-c", command)
    }

    try {
      osCommand.!!
    } catch {
      case e: RuntimeException if attempt < maxRetries =>
        println(s"Retrying after error: $e")
        Thread.sleep(1000)
        exec(command, maxRetries, attempt + 1)
    }
  }

  // Keep overhead of setting account down
  lazy val accountString: String = {
    try {
      exec(s"az account set -s $SubscriptionID")
    } catch {
      case e: java.lang.RuntimeException =>
        println(s"Secret fetch error: ${e.toString}")
      case e: IOException =>
        println(s"Secret fetch error: ${e.toString}")
    }
    SubscriptionID
  }

  private def getKeyvaultSecret(secretName: String): String = {
    println(s"[info] fetching secret: $secretName from $accountString")
    try {
      val secretJson = exec(s"az keyvault secret show --vault-name $KvName --name $secretName")
      secretJson.parseJson.asJsObject().fields("value").convertTo[String]
    } catch {
      case _: IOException =>
        println("WARNING: Could not load secret from keyvault, defaulting to the empty string." +
          " Please install az command line to perform authorized build steps like publishing")
        ""
      case _: java.lang.RuntimeException =>
        println("WARNING: Could not load secret from keyvault, defaulting to the empty string." +
          " Please install az command line to perform authorized build steps like publishing")
        ""
    }
  }

  private lazy val cacheDir: File = {
    val file = new File(".").getCanonicalPath
    val dir = BuildUtils.join(file, ".cache")
    if (!dir.exists()) {
      dir.mkdir()
    }
    dir
  }

  private def cachedFile(name: String): File = {
    BuildUtils.join(cacheDir, name)
  }

  private def getSecretFromCacheOrKeyvault(name: String): String = {
    val cachedSecretFile = cachedFile(name)
    if (cachedSecretFile.exists()) {
      println(s"[info] using cached value for secret $name.")
      val i = Source.fromFile(cachedSecretFile)
      try i.mkString finally i.close
    } else {
      println(s"[warn] could not find cached file or env var for secret $name. Fetching...")
      getKeyvaultSecret(name)
    }
  }

  /*
     Priority order for finding secrets:
     1. Environment variable (used by pipeline which loads them from keyvault)
     2. Local cache
     3. Load from keyvault
   */
  private def findAndCacheSecret(env_var: String, name: String): String = {
    sys.env.getOrElse(env_var, getSecretFromCacheOrKeyvault(name))
  }

  def getSecret(env_var: String, name: String): String = {
    if (publishingEnabled) findAndCacheSecret(env_var, name)
    else {
      println(s"[warn] Secret $name not downloaded. Set $EnablePublishEnvVar=true to enable publishing.")
      ""
    }
  }

  lazy val adoFeedToken: String = getSecret(ADOFeedTokenEnvVarName, ADOFeedTokenSecretName)
  lazy val pypiApiToken: String = getSecret(PypiApiEnvVarName, PypiApiSecretName)

  val ADOFeedTokenSecretName: String = "ado-feed-token"
  val ADOFeedTokenEnvVarName: String = "ADO-FEED-TOKEN"
  val PypiApiSecretName: String = "pypi-api-token"
  val PypiApiEnvVarName: String = "PYPI-API-TOKEN"
}
