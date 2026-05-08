# Lab orchestration (Cursor SDK)

## Dispatch a cloud worker

```bash
cd orchestration
npm install
export CURSOR_API_KEY=...
export OFFLINE_TARTEEL_REPO_URL=https://github.com/your-org/offline-tarteel-lab.git
export OFFLINE_TARTEEL_REF=main
npx tsx src/dispatch.ts "Your experiment prompt — edit tracker thresholds, run Tier 1, etc."
```

Optional:

- `CURSOR_AUTO_PR=1` — open PR from cloud agent
- `CURSOR_SKIP_REVIEWER=0` — allow reviewer request

## Judge (metrics JSON → accept/reject)

```bash
echo '{"target_recall":0.9,"baseline_recall":0.85,"blind_recall":0.88,"blind_baseline_recall":0.87,"onnx_mb":131,"tier3_completed":true}' \
  | npx tsx src/judgeCli.ts
```

Runs `lab_tools.judge_policy` in the lab venv.
