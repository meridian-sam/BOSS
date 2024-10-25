package io.epsilon3.plugin;

import java.util.List;
import java.util.Map;

public class TelemetryParameter {

    private String name;
    private String type;
    private String description;
    private Map<Integer, String> enumMapping;
    private List<TelemetryParameter> aggregateParameters;

    public TelemetryParameter(String name, String type, String description, Map<Integer, String> enumMapping, List<TelemetryParameter> aggregateParameters) {
      this.name = name;
        this.type = type;
        this.description = description;
        this.enumMapping = enumMapping;
        this.aggregateParameters = aggregateParameters;
    }

    public TelemetryParameter(String name, String type, String description) {
        this(name, type, description, null, null);
    }

    public TelemetryParameter(String name, String description, Map<Integer, String> enumMapping) {
        this(name, "enum", description, enumMapping, null);
    }

    public TelemetryParameter(String name, String description, List<TelemetryParameter> aggregateParameters) {
        this(name, "aggregate", description, null, aggregateParameters);
    }

    public String getName() {
        return name;
    }

    public String getType() {
        return type;
    }

    public String getDescription() {
        return description;
    }

    public Map<Integer, String> getEnumMapping() {
        return enumMapping;
    }

    public List<TelemetryParameter> getAggregateParameters() {
        return aggregateParameters;
    }

    public String toString() {
        return "Name: " + this.getName() + ". Type: " + this.getType() + ". Description: " + this.getDescription();
    }
}
