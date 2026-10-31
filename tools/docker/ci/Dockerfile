# CI image for building and testing the LightGBM module with sbt.
#
#   docker build -t synapseml-ci tools/docker/ci
#   docker run --rm -v "$PWD":/workspace synapseml-ci sbt -batch lightgbm/Test/compile
#
# JDK_VERSION: 11 for the Scala 2.12 branches (v1.0.8-spark3.5, spark3.5),
#              17 for the Spark 4 / Scala 2.13 branches (spark4.0, spark4.1).
ARG JDK_VERSION=11
FROM eclipse-temurin:${JDK_VERSION}-jdk-jammy

ARG SBT_VERSION=1.10.11
ENV SBT_OPTS="-Xmx4g -Dsbt.color=false -Dsbt.supershell=false"

# libgomp1: OpenMP runtime required by the native LightGBM library
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://github.com/sbt/sbt/releases/download/v${SBT_VERSION}/sbt-${SBT_VERSION}.tgz" \
      | tar xz -C /opt \
    && ln -s /opt/sbt/bin/sbt /usr/local/bin/sbt

WORKDIR /workspace
CMD ["sbt", "-batch", "lightgbm/Test/compile"]
