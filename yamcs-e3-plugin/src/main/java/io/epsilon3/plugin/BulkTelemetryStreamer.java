package io.epsilon3.plugin;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import io.socket.client.Socket;

import org.json.JSONArray;
import org.json.JSONObject;

import io.socket.client.Ack;

public class BulkTelemetryStreamer {

    private ConcurrentHashMap<String, ScheduledFuture<?>> scheduledTasks = new ConcurrentHashMap<>();
    private final ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(Runtime.getRuntime().availableProcessors() * 2); 

    private final Map<String, String> dictionaryIdToInstanceMap;
    private final Socket socket;

    public BulkTelemetryStreamer(Map<String, String> dictionaryIdToInstanceMap, Socket socket) {
        this.dictionaryIdToInstanceMap = dictionaryIdToInstanceMap;
        this.socket = socket;
    }

    public void onStartTelemetryStreaming(Object[] args) {
        if (args.length == 0 || !(args[0] instanceof JSONObject) || !(args[1] instanceof Ack)) {
            return;
        }

        JSONObject request = (JSONObject) args[0];
        String streamId = request.optString("stream_id");
        System.out.println("Received 'start_stream' event for " + request.optString("name"));

        if (!dictionaryIdToInstanceMap.containsKey(request.optString("dictionary_id"))) {
            return;
        }

        Map<String, Object> payload = new HashMap<>();
        payload.put("name", request.optString("name"));
        payload.put("operation", request.optString("operation"));
        payload.put("variables", request.opt("variables"));
        payload.put("metadata", request.opt("metadata"));
        payload.put("dictionary_id", request.optString("dictionary_id"));

        int refreshRate = request.optInt("refresh_rate", 1);
        startTelemetryStreaming(payload, streamId, refreshRate);
    }

    public void onEndTelemetryStreaming(Object[] args) {
        if (args.length == 0 || !(args[0] instanceof JSONObject) || !(args[1] instanceof Ack)) {
            return;
        }
        System.out.println("Received 'end_stream' event");

        JSONObject request = (JSONObject) args[0];
        try {
            JSONArray streamIds = request.getJSONArray("stream_ids");
            for (int i = 0; i < streamIds.length(); i++) {
                String streamId = streamIds.getString(i);
                cancelAndRemoveTask(streamId);
            }
        } catch (Exception e) {
            System.err.println("An error occurred removing streams: " + e.getMessage());
        }
    }

    public void onDisconnectTelemetryStream() {
        for (ScheduledFuture<?> task : scheduledTasks.values()) {
            if (task != null) {
                task.cancel(false);
            }
        }
        scheduledTasks.clear();
    }

    private synchronized void cancelAndRemoveTask(String streamId) {
        ScheduledFuture<?> task = scheduledTasks.get(streamId);
        if (task != null) {
            task.cancel(false);
            scheduledTasks.remove(streamId);
        }
    }

    private synchronized void startTelemetryStreaming(Map<String, Object> payload, String streamId, int refreshRate) {
        if (scheduledTasks.containsKey(streamId)) {
            System.out.println("Stream " + streamId + " is already running.");
            return;
        }
        
        Runnable sendDataUpdate = () -> {
            JSONObject sample = bulkTelemetryResponse(payload);
            try {
                JSONObject response = new JSONObject();
                response.put("data", sample);
                response.put("stream_id", streamId);
        
                socket.emit("data_update", response);
            } catch (Exception e) {
                System.err.println("failed to send data_update event for streamId: " + streamId);
            }
        };

        ScheduledFuture<?> task = executor.scheduleAtFixedRate(sendDataUpdate, 0, refreshRate, TimeUnit.SECONDS);
        scheduledTasks.put(streamId, task);
    }

    private JSONObject bulkTelemetryResponse(Map<String, Object> payload) {
        String parameterName = payload.get("name").toString();
        String dictionaryId = payload.get("dictionary_id").toString();
        String instanceName = dictionaryIdToInstanceMap.get(dictionaryId);
        try {
            JSONObject sample = TelemetryUtils.formatSampleReturn(instanceName, parameterName);
            return sample;
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        return null;
    }
}
