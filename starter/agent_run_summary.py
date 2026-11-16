"""Operational summary for agent traces; does not grade answer quality."""
import statistics


def summarize_batch(cases: list[dict], traces: list[dict], config: dict | None = None,
                    split: str | None = None, run_config: dict | None = None) -> dict:
    statuses = {name: sum(trace.get("status") == name for trace in traces)
                for name in ("answered", "abstain", "error")}
    successful = [trace for trace in traces if trace.get("status") in ("answered", "abstain")]
    searches = [sum(event.get("event") == "search" for event in trace.get("events", []))
                for trace in successful]
    group_summary = {}
    for answerable, label in ((True, "answerable"), (False, "unanswerable")):
        ids = {case["id"] for case in cases if case.get("answerable") is answerable}
        group = [trace for trace in traces if trace.get("query_id") in ids]
        group_successes = [trace for trace in group if trace.get("status") in ("answered", "abstain")]
        abstentions = sum(trace.get("status") == "abstain" for trace in group_successes)
        group_summary[label] = {
            "total": len(group),
            "answered": sum(trace.get("status") == "answered" for trace in group),
            "abstained": sum(trace.get("status") == "abstain" for trace in group),
            "errors": sum(trace.get("status") == "error" for trace in group),
            "abstention_rate_on_successes": abstentions / len(group_successes) if group_successes else None,
        }

    durations = [float(trace["elapsed_ms"]) for trace in traces if trace.get("elapsed_ms") is not None]
    total_citations = sum(len(trace.get("citations", [])) for trace in traces)
    llm_calls = [event for trace in traces for event in trace.get("events", [])
                 if event.get("event") == "llm_call"]
    reported_costs = [event["usage"].get("cost") if isinstance(event.get("usage"), dict) else None
                      for event in llm_calls]
    cost_usd = (sum(reported_costs) if llm_calls
                and all(isinstance(cost, (int, float)) for cost in reported_costs) else None)
    cases_with_search = sum(count > 0 for count in searches)
    return {
        "split": split,
        "retrieval_config": config or {},
        "run_config": run_config or {},
        "cases": len(traces),
        "status_counts": statuses,
        "by_answerability": group_summary,
        "retrieval_usage": {
            "cases_with_search": cases_with_search,
            "successful_cases": len(successful),
            "case_use_rate": cases_with_search / len(successful) if successful else None,
            "search_calls_total": sum(searches),
            "search_calls_per_successful_case_mean": statistics.mean(searches) if searches else 0,
        },
        "latency_ms": {
            "measured_cases": len(durations),
            "mean": statistics.mean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "total": sum(durations) if durations else None,
        },
        "citations": {
            "total_returned": total_citations,
            "answered_cases_with_citations": sum(
                trace.get("status") == "answered" and bool(trace.get("citations")) for trace in traces
            ),
        },
        "error_case_ids": [trace.get("query_id") for trace in traces if trace.get("status") == "error"],
        "quality_note": "Answer grades and semantic citation support require review labels; they are not inferred by this run summary.",
        "cost_usd": cost_usd,
        "cost_note": ("Sum of OpenRouter-reported usage.cost values. Null means one or more calls did not include cost; reconcile with OpenRouter Activity."),
    }
