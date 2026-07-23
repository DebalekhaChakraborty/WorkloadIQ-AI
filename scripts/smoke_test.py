from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
PORT = 8091
BASE_URL = f"http://127.0.0.1:{PORT}"


def wait_for_service(timeout_seconds: int = 70) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=1)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("Service did not become healthy before the timeout.")


def wait_for_assessment(job_id: str, timeout_seconds: int = 90) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(f"{BASE_URL}/api/assessments/{job_id}", timeout=5)
        response.raise_for_status()
        payload = response.json()
        if payload["status"] == "completed":
            return payload
        if payload["status"] == "failed":
            raise RuntimeError(payload.get("error") or "Assessment failed.")
        time.sleep(0.6)
    raise RuntimeError("Assessment did not complete before the timeout.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="workloadiq-smoke-") as output_dir:
        log_path = Path(output_dir) / "service.log"
        env = os.environ.copy()
        env.update(
            {
                "GOOGLE_CLOUD_PROJECT": "",
                "GOOGLE_CLOUD_LOCATION": "",
                "NORMALIZE_USE_LLM": "false",
                "TICKET_QA_OUTPUT_DIR": output_dir,
                "MPLCONFIGDIR": "/tmp/workloadiq-matplotlib",
                "PORT": str(PORT),
            }
        )
        with log_path.open("w", encoding="utf-8") as service_log:
            process = subprocess.Popen(
                [sys.executable, "-m", "workload_analysis.server_with_upload"],
                cwd=ROOT,
                env=env,
                stdout=service_log,
                stderr=subprocess.STDOUT,
            )
            try:
                wait_for_service()

                index = requests.get(f"{BASE_URL}/", timeout=5)
                index.raise_for_status()
                assert "WorkloadIQ" in index.text

                sample_path = ROOT / "sample_data" / "service_desk_tickets.csv"
                with sample_path.open("rb") as sample_file:
                    created = requests.post(
                        f"{BASE_URL}/api/assessments",
                        headers={"Origin": "http://10.128.0.2:3000"},
                        data={"mode": "workload"},
                        files={"file": (sample_path.name, sample_file, "text/csv")},
                        timeout=10,
                    )
                created.raise_for_status()
                job_id = created.json()["job_id"]
                assessment = wait_for_assessment(job_id)

                result = assessment["result"]
                assert result["total_tickets"] == 12
                assert len(result["top_categories"]) >= 4
                assert result["top_assignment_groups"]

                assistant = requests.post(
                    f"{BASE_URL}/api/copilot/chat",
                    json={"job_id": job_id, "query": "Summarize the main risk"},
                    timeout=10,
                )
                assistant.raise_for_status()
                assert assistant.json()["grounded"] is True
                assert assistant.json()["answer"]

                export = requests.get(
                    f"{BASE_URL}/api/download",
                    params={"job_id": job_id},
                    timeout=10,
                )
                export.raise_for_status()
                assert export.headers["content-type"].startswith("text/csv")

                print(
                    "Smoke test passed:",
                    {
                        "job_id": job_id,
                        "tickets": result["total_tickets"],
                        "categories": len(result["top_categories"]),
                        "network_origin_upload": "accepted",
                        "assistant": "grounded",
                        "export": "text/csv",
                    },
                )
            except Exception:
                service_log.flush()
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                if log_text:
                    print("Service log:\n" + log_text[-5000:], file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
