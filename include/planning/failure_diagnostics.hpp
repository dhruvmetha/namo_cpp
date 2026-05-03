#pragma once

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace namo {

struct FailureTraceEvent {
    std::string source = "unknown";
    std::string stage = "unknown";
    std::string code = "unknown";
    std::string message;
    int step_index_1based = 0;
    int edge_idx = -1;
    int push_steps = 0;
    int controller_stuck_counter = 0;
};

struct FailureDiagnostics {
    int schema_version = 1;
    std::string source = "unknown";
    std::string stage = "unknown";
    std::string code = "unknown";
    std::string summary;
    std::string detail;
    std::string collision_object;
    int step_index_1based = 0;
    int edge_idx = -1;
    int push_steps = 0;
    int controller_stuck_counter = 0;
    std::string nav_reason;
    int nav_steps_used = 0;
    std::vector<FailureTraceEvent> trace_events;

    void clear() { *this = FailureDiagnostics(); }

    bool has_signal() const {
        return !summary.empty() || !detail.empty() || code != "unknown" ||
               stage != "unknown" || source != "unknown";
    }

    void add_trace_event(const FailureTraceEvent& ev) { trace_events.push_back(ev); }

    std::map<std::string, std::string> to_flat_kv() const {
        std::map<std::string, std::string> out;
        out["failure_source"] = source;
        out["failure_stage"] = stage;
        out["failure_code"] = code;
        out["failure_detail"] = detail;
        out["failure_step_index"] = std::to_string(step_index_1based);
        out["failure_edge_idx"] = std::to_string(edge_idx);
        out["failure_push_steps"] = std::to_string(push_steps);
        out["failure_controller_stuck_counter"] = std::to_string(controller_stuck_counter);
        out["failure_nav_reason"] = nav_reason;
        out["failure_nav_steps_used"] = std::to_string(nav_steps_used);
        return out;
    }

    static std::string json_escape(const std::string& s) {
        std::ostringstream out;
        for (const unsigned char c : s) {
            switch (c) {
                case '\"': out << "\\\""; break;
                case '\\': out << "\\\\"; break;
                case '\b': out << "\\b"; break;
                case '\f': out << "\\f"; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default:
                    if (c < 0x20) {
                        static const char* kHex = "0123456789abcdef";
                        out << "\\u00" << kHex[(c >> 4) & 0x0f] << kHex[c & 0x0f];
                    } else {
                        out << static_cast<char>(c);
                    }
            }
        }
        return out.str();
    }

    std::string to_json(bool include_trace = false, int trace_max_events = 128) const {
        std::ostringstream out;
        out << "{";
        out << "\"schema_version\":" << schema_version;
        out << ",\"source\":\"" << json_escape(source) << "\"";
        out << ",\"stage\":\"" << json_escape(stage) << "\"";
        out << ",\"code\":\"" << json_escape(code) << "\"";
        out << ",\"summary\":\"" << json_escape(summary) << "\"";
        out << ",\"detail\":\"" << json_escape(detail) << "\"";
        out << ",\"collision_object\":\"" << json_escape(collision_object) << "\"";
        out << ",\"step_index_1based\":" << step_index_1based;
        out << ",\"edge_idx\":" << edge_idx;
        out << ",\"push_steps\":" << push_steps;
        out << ",\"controller_stuck_counter\":" << controller_stuck_counter;
        out << ",\"nav_reason\":\"" << json_escape(nav_reason) << "\"";
        out << ",\"nav_steps_used\":" << nav_steps_used;
        if (include_trace) {
            const int clamped_max = std::max(0, trace_max_events);
            const size_t n = std::min(trace_events.size(), static_cast<size_t>(clamped_max));
            out << ",\"trace_events\":[";
            for (size_t i = 0; i < n; ++i) {
                const auto& ev = trace_events[i];
                if (i > 0) out << ",";
                out << "{";
                out << "\"source\":\"" << json_escape(ev.source) << "\"";
                out << ",\"stage\":\"" << json_escape(ev.stage) << "\"";
                out << ",\"code\":\"" << json_escape(ev.code) << "\"";
                out << ",\"message\":\"" << json_escape(ev.message) << "\"";
                out << ",\"step_index_1based\":" << ev.step_index_1based;
                out << ",\"edge_idx\":" << ev.edge_idx;
                out << ",\"push_steps\":" << ev.push_steps;
                out << ",\"controller_stuck_counter\":" << ev.controller_stuck_counter;
                out << "}";
            }
            out << "]";
        }
        out << "}";
        return out.str();
    }
};

}  // namespace namo

