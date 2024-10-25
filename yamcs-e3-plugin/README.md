# Yamcs Plugin QuickStart

This repo demonstrates a basic plugin on Yamcs that integrates with Epsilon3.

## Prerequisites

- Java 11+
- Maven 3.1+
- Linux x64/aarch64, macOS x64, or Windows x64

## Steps to add the Epsilon3 Yamcs Plugin to Yamcs

1. `cd yamcs-plugin`
2. run `mvn clean package`
3. A jar file should have been generated in in `/target`. We need to install this jar to the local maven repository. Eventually, this would be published to the actual one. Run the following in the same dir as the jar file:

```
mvn install:install-file \
    -Dfile=yamcs-epsilon3-plugin-1.0.1.jar \
    -DgroupId=io.epsilon3.plugin \
    -DartifactId=yamcs-epsilon3-plugin \
    -Dversion=1.0.1 \
    -Dpackaging=jar
```

4. In your yamcs repo, open the pom.xml and add the following dependency:

```
<dependency>
  <groupId>io.epsilon3.plugin</groupId>
  <artifactId>yamcs-epsilon3-plugin</artifactId>
  <version>1.0.1</version>
</dependency>
```

5. This plugin requires the Epsilon3 Api Key be set as an environment variable where Yamcs is running. You can generate an API key in Epsilon3 settings
6. `export EPSILON3_API_KEY="key_abc"`
7. (Optional) By default, the Yamcs plugin uses the bulk telemetry routes. If you want to use the polling telemetry routes, set the following environment variable:
   `export BULK_TELEMETRY="false"`
8. Run or build Yamcs. You should see the Epsilon3 Yamcs plugin listed as a plugin under `Admin Area` -> `System`
9. All of the telemetry parameters in Yamcs should have been synced to Epsilon3. If you create a procedure with telemetry, those parameters should be available.
10. Running a procedure with Yamcs telemetry parameters will be updated in real-time with values from Yamcs.
