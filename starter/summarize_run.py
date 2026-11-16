"""Create an operational summary from an existing agent trace without rerunning it."""
import argparse
import json
from pathlib import Path

from agent_run_summary import summarize_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [case for case in cases if case.get("split") == args.split]
    if not cases:
        raise ValueError(f"No cases in split {args.split!r}: {args.cases}")
    trace_data = json.loads(args.traces.read_text(encoding="utf-8"))
    if isinstance(trace_data, dict):
        traces = trace_data.get("traces")
        config = trace_data.get("retrieval_config", {})
        run_config = trace_data.get("run_config", {})
        trace_split = trace_data.get("split")
        if trace_split and trace_split != args.split:
            raise ValueError(f"Trace file is split {trace_split!r}, not {args.split!r}")
    else:
        traces, config, run_config = trace_data, {}, {}
    if not isinstance(traces, list):
        raise ValueError("Trace file must be a JSON array or an object with a traces array")
    case_ids = {case["id"] for case in cases}
    traces = [trace for trace in traces if trace.get("query_id") in case_ids]
    trace_ids = [trace.get("query_id") for trace in traces]
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("Trace file contains duplicate query_id values for this split")
    missing = case_ids - set(trace_ids)
    if missing:
        raise ValueError(f"Trace file is missing cases: {sorted(missing)}")
    summary = summarize_batch(cases, traces, config, args.split, run_config)
    output = args.output or args.traces.with_name(f"{args.traces.stem}-summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Run summary saved to: {output}")


if __name__ == "__main__":
    main()
