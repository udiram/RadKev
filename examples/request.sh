#!/usr/bin/env bash
# The same request over HTTP, against Kev's own server (TypeSafe System One API):
#   python -m kev.serve --run <RadKev checkpoint> --port 8009
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' \
  -d "$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); r["model"]="kev-latest"; print(json.dumps(r))' "$(dirname "$0")/cxr_report.json")"
