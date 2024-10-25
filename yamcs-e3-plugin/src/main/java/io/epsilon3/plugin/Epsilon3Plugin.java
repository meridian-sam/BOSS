package io.epsilon3.plugin;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.json.JSONObject;
import org.yamcs.Plugin;
import org.yamcs.PluginException;
import org.yamcs.YConfiguration;
import org.yamcs.YamcsServer;
import org.yamcs.YamcsServerInstance;
import org.yamcs.mdb.MdbFactory;
import org.yamcs.xtce.AggregateParameterType;
import org.yamcs.xtce.DataType;
import org.yamcs.xtce.EnumeratedParameterType;
import org.yamcs.xtce.Member;
import org.yamcs.xtce.Parameter;
import org.yamcs.xtce.ParameterType;
import org.yamcs.xtce.ValueEnumeration;
import org.yamcs.xtce.XtceDb;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.socket.client.Ack;
import io.socket.client.IO;
import io.socket.client.Socket;
import io.socket.emitter.Emitter;

public class Epsilon3Plugin implements Plugin {

    private final String apiKey;
    private final String apiUrl;
    private final Epsilon3ApiClient epsilon3ApiClient;

    private final Map<String, String> dictionaryIdToInstanceMap;

    public Epsilon3Plugin() {
        apiKey = System.getenv("EPSILON3_API_KEY");
        apiUrl = System.getenv().getOrDefault("EPSILON3_API_URL", "https://api.epsilon3.io");

        epsilon3ApiClient = new Epsilon3ApiClient(apiUrl, apiKey);

        dictionaryIdToInstanceMap = new HashMap<>();
    }

    @Override
	public void onLoad(YConfiguration config) throws PluginException {
        if (apiKey == null) {
            throw new PluginException("The EPSILON3_API_KEY environment variable is not set! This must be set to use the plugin.");
        }

        System.out.println("Loading Epsilon3 Plugin");
        try {
            List<YamcsServerInstance> instances = YamcsServer.getInstances();
            for (YamcsServerInstance instance : instances) {
                setYamcsParametersInEpsilon3(instance.getName());
            }
            setupSocketIOConnection();
        } catch (Exception e) {
            System.out.println("An error occurred running the Epsilon3 plugin: " + e.getMessage());
        }
    }

    private void setYamcsParametersInEpsilon3(String instance) {
        String dictionaryId = getInstanceDictionaryId(instance);
        if (dictionaryId == null) {
            return;
        }
        dictionaryIdToInstanceMap.put(dictionaryId, instance);

        List<String> parameterNames = getEpsilon3TelemetryParamsForInstance(dictionaryId);
        if (parameterNames == null) {
            return;
        }
        
        // Find the Yamcs params that don't exist yet in Epsilon3
        Set<Parameter> uniqueParameterInfos = getUniqueYamcsParameters(instance, parameterNames);
        createYamcsParametersInEpsilon3(dictionaryId, uniqueParameterInfos);
    }

    private String getInstanceDictionaryId(String instance) {
        String dictionaries = epsilon3ApiClient.getDictionaries();
        ObjectMapper objectMapper = new ObjectMapper();
        String dictionaryId = null;
        List<Map<String, Object>> dictionaryObjects;
        try {
            dictionaryObjects = objectMapper.readValue(dictionaries, new TypeReference<List<Map<String, Object>>>() {});
        } catch (IOException e) {
            System.err.println("Error making request: " + e.getMessage());
            return null;
        }
        for (Map<String, Object> object : dictionaryObjects) {
            if (object.get("name").toString().equals(instance)) {
                dictionaryId = object.get("id").toString();
                break;
            }
        }
        if (dictionaryId == null) {
            dictionaryId = epsilon3ApiClient.createDictionary(instance);
        }
        return dictionaryId;
    }

    private List<String> getEpsilon3TelemetryParamsForInstance(String dictionaryId) {
        ObjectMapper objectMapper = new ObjectMapper();
        String telemetryParams = epsilon3ApiClient.getTelemetryParameters();
        List<Map<String, Object>> telemetryObjects;
        try {
            telemetryObjects = objectMapper.readValue(telemetryParams, new TypeReference<List<Map<String, Object>>>() {});
        } catch (IOException e) {
            System.err.println("Error making request: " + e.getMessage());
            return null;
        }

        // Find all the parameters that are in the Yamcs instance dictionary
        List<String> parameterNames = new ArrayList<>();
        for (Map<String, Object> object : telemetryObjects) {
            if (object.get("dictionary_id").toString().equals(dictionaryId)) {
                parameterNames.add(object.get("name").toString());
            }
        }
        return parameterNames;
    }

    private Set<Parameter> getUniqueYamcsParameters(String instance, List<String> parameterNames) {
        List<Parameter> yamcsParameters = getYamcsParameters(instance);
        Set<Parameter> uniqueParameters = new HashSet<Parameter>();
        for (Parameter parameter : yamcsParameters) {
            String parameterName = parameter.getQualifiedName();
            if (!parameterNames.contains(parameterName)) {
                uniqueParameters.add(parameter);
            }
        }
        return uniqueParameters;
    }

    private void createYamcsParametersInEpsilon3(String dictionaryId, Set<Parameter> uniqueParameters) {
        for (Parameter parameter : uniqueParameters) {
            String parameterType = mapYamcsDataTypeToEpsilon3(TelemetryUtils.getParameterTypeFromParameter(parameter));
            if (parameterType == null) {
                continue;
            }

            TelemetryParameter telemParam;
            String parameterName = parameter.getQualifiedName();
            String description = parameter.getShortDescription();
            
            if ("enum".equals(parameterType)) {
              Map<Integer, String> enumMapping = buildEnumMapping((EnumeratedParameterType) parameter.getParameterType());
              telemParam = new TelemetryParameter(parameterName, description, enumMapping);
            } else if ("aggregate".equals(parameterType)) {
              // First create TelemetryParameters for each of the parameters defined in the aggregate
                ParameterType ptype = parameter.getParameterType();
                AggregateParameterType agtype = (AggregateParameterType) ptype;
                List<Member> aggregateParameters = agtype.getMemberList();

                List<TelemetryParameter> aggregateTelemetryParameters = new ArrayList<>();
                for (Member aggregateParameter : aggregateParameters) {
                    DataType dtype = aggregateParameter.getType();
                    String aggregateParameterType = mapYamcsDataTypeToEpsilon3(dtype.getTypeAsString());
                    
                    if ("enum".equals(aggregateParameterType)) {
                        Map<Integer, String> enumMapping = buildEnumMapping((EnumeratedParameterType) dtype);
                        aggregateTelemetryParameters.add(new TelemetryParameter(aggregateParameter.getName(), "", enumMapping));
                    } else {
                        aggregateTelemetryParameters.add(new TelemetryParameter(aggregateParameter.getName(), aggregateParameterType, ""));
                    }
                }
                telemParam = new TelemetryParameter(parameterName, description, aggregateTelemetryParameters);
            }
            else {
                telemParam = new TelemetryParameter(parameterName, parameterType, description);
            }

            String response = epsilon3ApiClient.createTelemetryParameter(telemParam, dictionaryId);
            if (response == null) {
                System.out.println("Failed to create telemetry parameter: " + telemParam);
            }
        }
    }

    private Map<Integer, String> buildEnumMapping(EnumeratedParameterType eptype) {
      List<ValueEnumeration> enumerations = eptype.getValueEnumerationList();
      Map<Integer, String> enumMapping = new HashMap<>();
      for(ValueEnumeration ve : enumerations) {
          long value = ve.getValue();
          String label = ve.getLabel();
          enumMapping.put((int) value, label);
      }
      return enumMapping;
  }

    private String mapYamcsDataTypeToEpsilon3(String parameterType) {
        if (parameterType.equals("integer")) {
            return "int";
        } else if (parameterType.equals("boolean")) {
            return "bool";
        } else if (parameterType.equals("enumeration")) {
            return "enum";
        }
        // "float", "string", and "aggregate" match the data type in the Epsilon3 telemetry endpoint
        return parameterType;
    }

    private void setupSocketIOConnection() {
        IO.Options options = IO.Options.builder()
                .setAuth(Collections.singletonMap("key", apiKey))
                .setTransports(new String[]{"websocket"})
                .build();

        boolean bulkEnabled = Boolean.parseBoolean(System.getenv().getOrDefault("BULK_TELEMETRY", "true"));

        try {
            // The Java SocketIO client requires a socket connection per namespace
            // If we want to add commanding or external data, would need to create a separate socket connection
            final Socket socket = IO.socket(URI.create(apiUrl + "/v1/telemetry/realtime"), options);

            socket.on(Socket.EVENT_CONNECT, args -> System.out.println("Connected to server"));
            socket.on(Socket.EVENT_CONNECT_ERROR, args -> System.out.println("Connection error: " + args[0]));

            if (bulkEnabled) {
                BulkTelemetryStreamer bulkTelemetryStreamer = new BulkTelemetryStreamer(dictionaryIdToInstanceMap, socket);
                socket.on("start_stream", args -> bulkTelemetryStreamer.onStartTelemetryStreaming(args));
                socket.on("end_streams", args -> bulkTelemetryStreamer.onEndTelemetryStreaming(args));
                socket.on(Socket.EVENT_DISCONNECT, args -> bulkTelemetryStreamer.onDisconnectTelemetryStream());
            } else {
                socket.on(Socket.EVENT_DISCONNECT, args -> System.out.println("Disconnect: " + args[0]));

                // Define the event listener for the "get_sample" event
                socket.on("get_sample", new Emitter.Listener() {
                    @Override
                    public void call(Object... args) {
                        System.out.println("Received 'get_sample' event");
    
                        if (args.length == 0 || !(args[0] instanceof JSONObject) || !(args[1] instanceof Ack)) {
                            return;
                        }
    
                        JSONObject jsonObject = (JSONObject) args[0];
                        Ack ack = (Ack) args[1];
    
                        String parameterName = jsonObject.optString("name");
                        String dictionaryId = jsonObject.optString("dictionaryId");
    
                        if (!dictionaryIdToInstanceMap.containsKey(dictionaryId)) {
                            return;
                        }
                        String instanceName = dictionaryIdToInstanceMap.get(dictionaryId);
    
                        try {
                            JSONObject sample = TelemetryUtils.formatSampleReturn(instanceName, parameterName);
    
                            ack.call(sample);
                        } catch (Exception e) {
                            System.out.println(e.getMessage());
                        }
                    }
                });
            }
            socket.connect();
        } catch (Exception e) {
            System.err.println("Error making request: " + e.getMessage());
        }
    }

    /*
     * Copied the core logic from the MDB API source code:
     * (https://github.com/yamcs/yamcs/blob/master/yamcs-core/src/main/java/org/yamcs/http/api/MdbApi.java#L212)
     * I removed the references to the space systems for now. If there's a need for them, can add it back
     */
    private List<Parameter> getYamcsParameters(String instance) {
        XtceDb mdb = MdbFactory.getInstance(instance);

        List<Parameter> allParameters = new ArrayList<>();
        mdb.getParameters().stream().forEach(parameter -> {
            allParameters.add(parameter);
        });

        return allParameters;
    }
}
