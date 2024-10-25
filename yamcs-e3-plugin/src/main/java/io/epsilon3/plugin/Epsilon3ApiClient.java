package io.epsilon3.plugin;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpClient.Version;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class Epsilon3ApiClient {

    private final String apiKey;
    private final String endpoint;
    private final HttpClient httpClient;

    private final String GET = "GET";
    private final String POST = "POST";

    private final String TELEMETRY_ENDPOINT = "/v1/telemetry/parameters";
    private final String DICTIONARY_ENDPOINT = "/v1/dictionaries";

    public Epsilon3ApiClient(String endpoint, String apiKey) {
        this.apiKey = apiKey;
        this.endpoint = endpoint;
        httpClient = HttpClient.newHttpClient();
    }

    private String makeRequest(String path) {
        return makeRequest(path, GET, null);
    }

    private String makeRequest(String path, String method, String jsonPayload) {
        try {
            String url = endpoint + path;
            String authHeader = "Basic " + Base64.getEncoder().encodeToString((apiKey + ":").getBytes());
    
            HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                    .version(Version.HTTP_1_1)
                    .uri(new URI(url))
                    .header("Authorization", authHeader)
                    .method(method, (jsonPayload != null) ? HttpRequest.BodyPublishers.ofString(jsonPayload) : HttpRequest.BodyPublishers.noBody());
            
            if ("POST".equals(method) && jsonPayload != null) {
                requestBuilder.header("Content-Type", "application/json");
            }
    
            HttpRequest request = requestBuilder.build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                System.err.println("Error occurred, status code: " + response.statusCode());
                return null;
            }
    
            return response.body();
        } catch (Exception e) {
            System.err.println("Error making request: " + e.getMessage());
        }
        return null;
    }

    public String getTelemetryParameters() {
        return makeRequest(TELEMETRY_ENDPOINT);
    }

    public String getDictionaries() {
        return makeRequest(DICTIONARY_ENDPOINT);
    }
    
    public String createDictionary(String dictionaryName) {
        ObjectMapper objectMapper = new ObjectMapper();
        Map<String, String> map = new HashMap<>();
        map.put("name", dictionaryName);
        try {
            String jsonPayload = objectMapper.writeValueAsString(map);
            String response = makeRequest(DICTIONARY_ENDPOINT, POST, jsonPayload);
            JsonNode rootNode = objectMapper.readTree(response);
            String id = rootNode.get("id").toString();
            return id;
        } catch (Exception e) {
            System.err.println("Error making request: " + e.getMessage());
        }
        return null;
    }

    public String createTelemetryParameter(TelemetryParameter parameter, String dictionaryId) {
        ObjectMapper objectMapper = new ObjectMapper();
        Map<String, Object> map = new HashMap<>();

        String parameterType = parameter.getType();
        map.put("name", parameter.getName());
        map.put("type", parameterType);
        map.put("dictionary_id", dictionaryId);

        if (parameter.getDescription() != null && !"".equals(parameter.getDescription())) {
            map.put("description", parameter.getDescription());
        }

        if ("enum".equals(parameterType)) {
            map.put("values", parameter.getEnumMapping());
        } else if ("aggregate".equals(parameterType)) {
            List<Map<String, Object>> aggregateValuesList = new ArrayList<>();
            for (TelemetryParameter subParam : parameter.getAggregateParameters()) {
                Map<String, Object> subParamMap = new HashMap<>();
                subParamMap.put("name", subParam.getName());
                String subParamType = subParam.getType();
                subParamMap.put("type", subParamType);
                if ("enum".equals(subParamType)) {
                    subParamMap.put("values", subParam.getEnumMapping());
                }
                aggregateValuesList.add(subParamMap);
            }
            map.put("values", aggregateValuesList);
        }
        try {
            String jsonPayload = objectMapper.writeValueAsString(map);
            return makeRequest(TELEMETRY_ENDPOINT, POST, jsonPayload);
        } catch (Exception e) {
            System.err.println("Error making request: " + e.getMessage());
        }
        return null;
    }
}
