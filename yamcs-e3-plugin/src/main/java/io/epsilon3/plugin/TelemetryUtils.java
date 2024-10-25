package io.epsilon3.plugin;

import java.time.Instant;

import org.json.JSONObject;
import org.yamcs.Processor;
import org.yamcs.YamcsServer;
import org.yamcs.YamcsServerInstance;
import org.yamcs.http.api.XtceToGpbAssembler;
import org.yamcs.http.api.XtceToGpbAssembler.DetailLevel;
import org.yamcs.parameter.AggregateValue;
import org.yamcs.parameter.ParameterRequestManager;
import org.yamcs.parameter.ParameterValue;
import org.yamcs.parameter.Value;
import org.yamcs.protobuf.Mdb.ParameterInfo;
import org.yamcs.protobuf.Mdb.ParameterTypeInfo;
import org.yamcs.xtce.Parameter;
import org.yamcs.xtce.util.AggregateMemberNames;

public class TelemetryUtils {
    private static final String REALTIME_PROCESSOR = "realtime";

    public static JSONObject formatSampleReturn(String instanceName, String parameterName) {
        YamcsServer yamcs = YamcsServer.getServer();
        YamcsServerInstance instance = yamcs.getInstance(instanceName);
        Processor processor = instance.getProcessor(TelemetryUtils.REALTIME_PROCESSOR);
        ParameterRequestManager prm = processor.getParameterRequestManager();

        JSONObject sample = new JSONObject();

        try {
            Parameter p = prm.getParameter(parameterName);
            ParameterValue pv = prm.getLastValueFromCache(p);

            String parameterType = getParameterTypeFromParameter(p);
            if ("enumeration".equals(parameterType)) {
                sample.put("value", pv.getRawValue());
            } else if ("aggregate".equals(parameterType)) {
                JSONObject aggregateJson = new JSONObject();

                AggregateValue aggrv = (AggregateValue) pv.getRawValue();
                AggregateMemberNames aggMembNames = aggrv.getMemberNames();
                for (int i = 0; i < aggMembNames.size(); i++) {
                    Value aggVal = aggrv.getMemberValue(i);
                    aggregateJson.put(aggrv.getMemberName(i), aggVal.toString());
                }
                sample.put("value", aggregateJson);
            } 
            else {
                sample.put("value", pv.getEngValue());
            }

            sample.put("recorded_at", Instant.ofEpochMilli(pv.getGenerationTime()).toString());
            sample.put("name", parameterName);
            sample.put("stale_after_ms", 1000);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        return sample;
    }

    public static String getParameterTypeFromParameter(Parameter parameter) {
        ParameterInfo parameterInfo = XtceToGpbAssembler.toParameterInfo(parameter, DetailLevel.SUMMARY);
        ParameterTypeInfo parameterTypeInfo = parameterInfo.getType();
        return parameterTypeInfo.getEngType();
    }
}
