#include "knob_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int json_get_int(const char* json, const char* key, int default_val) {
    char search[300];
    snprintf(search, sizeof(search), "\"%s\":", key);

    const char* pos = strstr(json, search);
    if (!pos) return default_val;

    pos += strlen(search);
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r')
        pos++;

    if (*pos == '\0') return default_val;
    return atoi(pos);
}

static char* read_file_to_string(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);

    if (sz <= 0) { fclose(f); return NULL; }

    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t read = fread(buf, 1, (size_t)sz, f);
    buf[read] = '\0';
    fclose(f);
    return buf;
}

static int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

void get_knobs(
    int* sim_id,
    int* knob_load_weight,
    int* knob_store_weight,
    int* knob_branch_weight,
    int* knob_jump_weight,
    int* knob_arith_weight,
    int* knob_mem_stride,
    int* knob_pointer_update_rate,
    int* knob_trap_rate,
    int* knob_trap_kind,
    int* knob_branch_taken_bias,
    int* knob_mixed_burst_bias,
    int* knob_mem_delay_base)
{
    const char* env_id = getenv("SIM_ID");
    *sim_id = (env_id != NULL) ? atoi(env_id) : 0;

    char knob_path[MAX_PATH];
    snprintf(knob_path, sizeof(knob_path), "%s/knobs_%d.json", WORK_DIR, *sim_id);

    fprintf(stdout, "[DPI] get_knobs: reading %s\n", knob_path);
    fflush(stdout);

    char* json = read_file_to_string(knob_path);

    if (json == NULL) {
        fprintf(stderr, "[DPI] WARNING: %s not found - using defaults\n", knob_path);
        *knob_load_weight    = 5;
        *knob_store_weight   = 5;
        *knob_branch_weight  = 3;
        *knob_jump_weight    = 3;
        *knob_arith_weight   = 5;
        *knob_mem_stride     = 4;
        *knob_pointer_update_rate = 3;
        *knob_trap_rate      = 0;
        *knob_trap_kind      = 0;
        *knob_branch_taken_bias = 5;
        *knob_mixed_burst_bias = 2;
        *knob_mem_delay_base = 3;
        return;
    }

    *knob_load_weight    = json_get_int(json, "load_weight",    5);
    *knob_store_weight   = json_get_int(json, "store_weight",   5);
    *knob_branch_weight  = json_get_int(json, "branch_weight",  3);
    *knob_jump_weight    = json_get_int(json, "jump_weight",    3);
    *knob_arith_weight   = json_get_int(json, "arith_weight",   5);
    *knob_mem_stride     = json_get_int(json, "mem_stride",     4);
    *knob_pointer_update_rate = json_get_int(json, "pointer_update_rate", 3);
    *knob_trap_rate      = json_get_int(json, "trap_rate",      0);
    *knob_trap_kind      = json_get_int(json, "trap_kind",      0);
    *knob_branch_taken_bias = json_get_int(json, "branch_taken_bias", 5);
    *knob_mixed_burst_bias = json_get_int(json, "mixed_burst_bias", 2);
    *knob_mem_delay_base = json_get_int(json, "mem_delay_base", 3);

    free(json);

    *knob_load_weight    = clamp(*knob_load_weight,    1, 10);
    *knob_store_weight   = clamp(*knob_store_weight,   1, 10);
    *knob_branch_weight  = clamp(*knob_branch_weight,  1, 10);
    *knob_jump_weight    = clamp(*knob_jump_weight,    1, 10);
    *knob_arith_weight   = clamp(*knob_arith_weight,   1, 10);
    *knob_mem_stride     = clamp(*knob_mem_stride,     1,  8);
    *knob_pointer_update_rate = clamp(*knob_pointer_update_rate, 1, 10);
    *knob_trap_rate      = clamp(*knob_trap_rate,      0,  3);
    *knob_trap_kind      = clamp(*knob_trap_kind,      0,  3);
    *knob_branch_taken_bias = clamp(*knob_branch_taken_bias, 0, 10);
    *knob_mixed_burst_bias = clamp(*knob_mixed_burst_bias, 0, 10);
    *knob_mem_delay_base = clamp(*knob_mem_delay_base, 1,  8);

    fprintf(stdout,
        "[DPI] Knobs: load=%d store=%d branch=%d jump=%d arith=%d stride=%d ptr_upd=%d trap_rate=%d trap_kind=%d br_bias=%d mix_bias=%d delay_base=%d\n",
        *knob_load_weight, *knob_store_weight, *knob_branch_weight,
        *knob_jump_weight, *knob_arith_weight, *knob_mem_stride,
        *knob_pointer_update_rate, *knob_trap_rate, *knob_trap_kind,
        *knob_branch_taken_bias, *knob_mixed_burst_bias, *knob_mem_delay_base);
    fflush(stdout);
}

void write_coverage(
    int sim_id,
    int total_cycles,
    int active_cycles,
    int instr_stall_cycles,
    int load_stall_cycles,
    int store_stall_cycles,
    int total_stall_cycles,
    int completed_accesses,
    int completed_instr_accesses,
    int completed_load_accesses,
    int completed_store_accesses,
    int max_stall_run,
    int stall_runs_gt2,
    int stall_runs_gt4,
    int stall_runs_gt8,
    int long_instr_stall_runs,
    int long_load_stall_runs,
    int long_store_stall_runs,
    int b2b_stall_count,
    int data_burst_count,
    int mixed_burst_count,
    int fetch_then_load,
    int fetch_then_store,
    int load_then_load,
    int load_then_store,
    int store_then_load,
    int store_then_store,
    int transition_types_hit,
    int mixed_data_transition_count,
    int consecutive_store_bursts,
    int consecutive_mixed_bursts,
    int instr_count,
    int load_instr_count,
    int store_instr_count,
    int branch_instr_count,
    int jump_instr_count,
    int arith_instr_count,
    int total_mem_requests,
    int instr_requests,
    int data_requests,
    int trap_count,
    int intermediate_trap_count,
    int recovery_count,
    int timed_out,
    int data_region_checksum,
    int bug_region_hit_count,
    int bug_arm_count,
    int bug_branch_candidate_count,
    int bug_manifest_count,
    int raw_mixed_transition_count,
    int raw_shortstall_data_xfer_count,
    int raw_taken_branch_count,
    int raw_taken_branch_while_armed_count,
    int knob_load_weight,
    int knob_store_weight,
    int knob_branch_weight,
    int knob_jump_weight,
    int knob_arith_weight,
    int knob_mem_stride,
    int knob_pointer_update_rate,
    int knob_trap_rate,
    int knob_trap_kind,
    int knob_branch_taken_bias,
    int knob_mixed_burst_bias,
    int knob_mem_delay_base,
    int near_trap_deep_stall,
    int near_mixed_b2b,
    int near_transition_diversity,
    int near_long_data_stall,
    int near_full_stress)
{
    char cov_path[MAX_PATH];
    snprintf(cov_path, sizeof(cov_path), "%s/coverage_%d.json", WORK_DIR, sim_id);

    FILE* f = fopen(cov_path, "w");
    if (!f) {
        fprintf(stderr, "[DPI] ERROR: Cannot write %s\n", cov_path);
        return;
    }

    double total_c     = (total_cycles > 0) ? (double)total_cycles : 1.0;
    double total_r     = (total_mem_requests > 0) ? (double)total_mem_requests : 1.0;
    double total_i     = (instr_count > 0) ? (double)instr_count : 1.0;
    double completed_r = (completed_accesses > 0) ? (double)completed_accesses : 1.0;

    double stall_ratio        = (double)total_stall_cycles / total_c;
    double instr_stall_ratio  = (double)instr_stall_cycles / total_c;
    double load_stall_ratio   = (double)load_stall_cycles / total_c;
    double store_stall_ratio  = (double)store_stall_cycles / total_c;
    double data_stall_ratio   = (double)(load_stall_cycles + store_stall_cycles) / total_c;
    double b2b_stall_rate     = (double)b2b_stall_count / total_r;
    double completed_b2b_rate = (double)b2b_stall_count / completed_r;
    double active_ratio       = (double)active_cycles / total_c;
    double mem_pressure       = (double)total_mem_requests / total_c;
    double completed_pressure = (double)completed_accesses / total_c;
    double load_instr_frac    = (double)load_instr_count / total_i;
    double store_instr_frac   = (double)store_instr_count / total_i;
    double branch_instr_frac  = (double)branch_instr_count / total_i;
    double jump_instr_frac    = (double)jump_instr_count / total_i;
    double total_data_transitions = (double)(load_then_load + load_then_store +
                                             store_then_load + store_then_store);
    double long_stall_mix     = (double)(long_instr_stall_runs +
                                         long_load_stall_runs +
                                         long_store_stall_runs) / completed_r;
    double load_store_alternation_ratio =
        (total_data_transitions > 0.0) ?
        ((double)mixed_data_transition_count / total_data_transitions) : 0.0;
    double transition_entropy = 0.0;
    double near_miss_score    = (double)(near_trap_deep_stall +
                                         near_mixed_b2b +
                                         near_transition_diversity +
                                         near_long_data_stall +
                                         near_full_stress) / 5.0;
    {
        double counts[6];
        counts[0] = (double)fetch_then_load;
        counts[1] = (double)fetch_then_store;
        counts[2] = (double)load_then_load;
        counts[3] = (double)load_then_store;
        counts[4] = (double)store_then_load;
        counts[5] = (double)store_then_store;
        double total_transitions = 0.0;
        int i;
        for (i = 0; i < 6; ++i)
            total_transitions += counts[i];
        if (total_transitions > 0.0) {
            double sum_sq = 0.0;
            for (i = 0; i < 6; ++i) {
                double p = counts[i] / total_transitions;
                sum_sq += p * p;
            }
            transition_entropy = 1.0 - sum_sq;
        }
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"sim_id\": %d,\n", sim_id);
    fprintf(f, "  \"coverage\": {\n");
    fprintf(f, "    \"total_cycles\": %d,\n", total_cycles);
    fprintf(f, "    \"active_cycles\": %d,\n", active_cycles);
    fprintf(f, "    \"instr_stall_cycles\": %d,\n", instr_stall_cycles);
    fprintf(f, "    \"load_stall_cycles\": %d,\n", load_stall_cycles);
    fprintf(f, "    \"store_stall_cycles\": %d,\n", store_stall_cycles);
    fprintf(f, "    \"total_stall_cycles\": %d,\n", total_stall_cycles);
    fprintf(f, "    \"completed_accesses\": %d,\n", completed_accesses);
    fprintf(f, "    \"completed_instr_accesses\": %d,\n", completed_instr_accesses);
    fprintf(f, "    \"completed_load_accesses\": %d,\n", completed_load_accesses);
    fprintf(f, "    \"completed_store_accesses\": %d,\n", completed_store_accesses);
    fprintf(f, "    \"stall_ratio\": %.6f,\n", stall_ratio);
    fprintf(f, "    \"instr_stall_ratio\": %.6f,\n", instr_stall_ratio);
    fprintf(f, "    \"load_stall_ratio\": %.6f,\n", load_stall_ratio);
    fprintf(f, "    \"store_stall_ratio\": %.6f,\n", store_stall_ratio);
    fprintf(f, "    \"data_stall_ratio\": %.6f,\n", data_stall_ratio);
    fprintf(f, "    \"active_ratio\": %.6f,\n", active_ratio);
    fprintf(f, "    \"mem_pressure\": %.6f,\n", mem_pressure);
    fprintf(f, "    \"completed_pressure\": %.6f,\n", completed_pressure);
    fprintf(f, "    \"max_stall_run\": %d,\n", max_stall_run);
    fprintf(f, "    \"stall_runs_gt2\": %d,\n", stall_runs_gt2);
    fprintf(f, "    \"stall_runs_gt4\": %d,\n", stall_runs_gt4);
    fprintf(f, "    \"stall_runs_gt8\": %d,\n", stall_runs_gt8);
    fprintf(f, "    \"long_instr_stall_runs\": %d,\n", long_instr_stall_runs);
    fprintf(f, "    \"long_load_stall_runs\": %d,\n", long_load_stall_runs);
    fprintf(f, "    \"long_store_stall_runs\": %d,\n", long_store_stall_runs);
    fprintf(f, "    \"long_stall_mix\": %.6f,\n", long_stall_mix);
    fprintf(f, "    \"b2b_stall_count\": %d,\n", b2b_stall_count);
    fprintf(f, "    \"b2b_stall_rate\": %.6f,\n", b2b_stall_rate);
    fprintf(f, "    \"completed_b2b_rate\": %.6f,\n", completed_b2b_rate);
    fprintf(f, "    \"data_burst_count\": %d,\n", data_burst_count);
    fprintf(f, "    \"mixed_burst_count\": %d,\n", mixed_burst_count);
    fprintf(f, "    \"fetch_then_load\": %d,\n", fetch_then_load);
    fprintf(f, "    \"fetch_then_store\": %d,\n", fetch_then_store);
    fprintf(f, "    \"load_then_load\": %d,\n", load_then_load);
    fprintf(f, "    \"load_then_store\": %d,\n", load_then_store);
    fprintf(f, "    \"store_then_load\": %d,\n", store_then_load);
    fprintf(f, "    \"store_then_store\": %d,\n", store_then_store);
    fprintf(f, "    \"transition_types_hit\": %d,\n", transition_types_hit);
    fprintf(f, "    \"mixed_data_transition_count\": %d,\n", mixed_data_transition_count);
    fprintf(f, "    \"consecutive_store_bursts\": %d,\n", consecutive_store_bursts);
    fprintf(f, "    \"consecutive_mixed_bursts\": %d,\n", consecutive_mixed_bursts);
    fprintf(f, "    \"load_store_alternation_ratio\": %.6f,\n", load_store_alternation_ratio);
    fprintf(f, "    \"transition_entropy\": %.6f,\n", transition_entropy);
    fprintf(f, "    \"instr_count\": %d,\n", instr_count);
    fprintf(f, "    \"load_instr_count\": %d,\n", load_instr_count);
    fprintf(f, "    \"store_instr_count\": %d,\n", store_instr_count);
    fprintf(f, "    \"branch_instr_count\": %d,\n", branch_instr_count);
    fprintf(f, "    \"jump_instr_count\": %d,\n", jump_instr_count);
    fprintf(f, "    \"arith_instr_count\": %d,\n", arith_instr_count);
    fprintf(f, "    \"load_instr_frac\": %.6f,\n", load_instr_frac);
    fprintf(f, "    \"store_instr_frac\": %.6f,\n", store_instr_frac);
    fprintf(f, "    \"branch_instr_frac\": %.6f,\n", branch_instr_frac);
    fprintf(f, "    \"jump_instr_frac\": %.6f,\n", jump_instr_frac);
    fprintf(f, "    \"total_mem_requests\": %d,\n", total_mem_requests);
    fprintf(f, "    \"instr_requests\": %d,\n", instr_requests);
    fprintf(f, "    \"data_requests\": %d,\n", data_requests);
    fprintf(f, "    \"trap_count\": %d,\n", trap_count);
    fprintf(f, "    \"intermediate_trap_count\": %d,\n", intermediate_trap_count);
    fprintf(f, "    \"recovery_count\": %d,\n", recovery_count);
    fprintf(f, "    \"timed_out\": %d,\n", timed_out);
    fprintf(f, "    \"trap_fired\": %s,\n", (trap_count > 0) ? "true" : "false");
    fprintf(f, "    \"data_region_checksum\": %u,\n", (unsigned int)data_region_checksum);
    fprintf(f, "    \"bug_region_hit_count\": %d,\n", bug_region_hit_count);
    fprintf(f, "    \"bug_arm_count\": %d,\n", bug_arm_count);
    fprintf(f, "    \"bug_branch_candidate_count\": %d,\n", bug_branch_candidate_count);
    fprintf(f, "    \"bug_manifest_count\": %d,\n", bug_manifest_count);
    fprintf(f, "    \"raw_mixed_transition_count\": %d,\n", raw_mixed_transition_count);
    fprintf(f, "    \"raw_shortstall_data_xfer_count\": %d,\n", raw_shortstall_data_xfer_count);
    fprintf(f, "    \"raw_taken_branch_count\": %d,\n", raw_taken_branch_count);
    fprintf(f, "    \"raw_taken_branch_while_armed_count\": %d,\n", raw_taken_branch_while_armed_count);
    fprintf(f, "    \"near_trap_deep_stall\": %d,\n", near_trap_deep_stall);
    fprintf(f, "    \"near_mixed_b2b\": %d,\n", near_mixed_b2b);
    fprintf(f, "    \"near_transition_diversity\": %d,\n", near_transition_diversity);
    fprintf(f, "    \"near_long_data_stall\": %d,\n", near_long_data_stall);
    fprintf(f, "    \"near_full_stress\": %d,\n", near_full_stress);
    fprintf(f, "    \"near_miss_score\": %.6f,\n", near_miss_score);
    fprintf(f, "    \"knobs\": {\n");
    fprintf(f, "      \"load_weight\": %d,\n", knob_load_weight);
    fprintf(f, "      \"store_weight\": %d,\n", knob_store_weight);
    fprintf(f, "      \"branch_weight\": %d,\n", knob_branch_weight);
    fprintf(f, "      \"jump_weight\": %d,\n", knob_jump_weight);
    fprintf(f, "      \"arith_weight\": %d,\n", knob_arith_weight);
    fprintf(f, "      \"mem_stride\": %d,\n", knob_mem_stride);
    fprintf(f, "      \"pointer_update_rate\": %d,\n", knob_pointer_update_rate);
    fprintf(f, "      \"trap_rate\": %d,\n", knob_trap_rate);
    fprintf(f, "      \"trap_kind\": %d,\n", knob_trap_kind);
    fprintf(f, "      \"branch_taken_bias\": %d,\n", knob_branch_taken_bias);
    fprintf(f, "      \"mixed_burst_bias\": %d,\n", knob_mixed_burst_bias);
    fprintf(f, "      \"mem_delay_base\": %d\n", knob_mem_delay_base);
    fprintf(f, "    }\n");
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);

    fprintf(stdout, "[DPI] Coverage written: %s\n", cov_path);
    fprintf(stdout,
        "[DPI] stall_ratio=%.4f data_stall=%.4f max_run=%d transition_types=%d alternation=%.3f entropy=%.3f intermediate_traps=%d timeout=%d near_miss=%.2f checksum=%u bug_region=%d bug_arm=%d bug_branch=%d bug_manifest=%d raw_mixed=%d raw_short=%d raw_taken=%d raw_taken_arm=%d\n",
        stall_ratio, data_stall_ratio, max_stall_run,
        transition_types_hit, load_store_alternation_ratio, transition_entropy,
        intermediate_trap_count, timed_out, near_miss_score, (unsigned int)data_region_checksum,
        bug_region_hit_count, bug_arm_count, bug_branch_candidate_count, bug_manifest_count,
        raw_mixed_transition_count, raw_shortstall_data_xfer_count,
        raw_taken_branch_count, raw_taken_branch_while_armed_count);
    fflush(stdout);
}
