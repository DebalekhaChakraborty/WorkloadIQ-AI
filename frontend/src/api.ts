import type { Assessment, AssessmentMode } from './types';

async function ensureOk(response: Response): Promise<Response> {
  if (response.ok) return response;
  const raw = await response.text().catch(() => '');
  let payload: { detail?: string; error?: string } | null = null;
  try {
    payload = raw ? JSON.parse(raw) : null;
  } catch {
    payload = null;
  }
  const message = payload?.detail || payload?.error || raw || response.statusText;
  throw new Error(String(message));
}

export const api = {
  async listAssessments(): Promise<Assessment[]> {
    const response = await ensureOk(await fetch('/api/assessments?limit=30'));
    const payload = await response.json();
    return payload.items || [];
  },

  async getAssessment(jobId: string): Promise<Assessment> {
    const response = await ensureOk(await fetch(`/api/assessments/${encodeURIComponent(jobId)}`));
    return response.json();
  },

  async createAssessment(file: File, mode: AssessmentMode): Promise<Assessment> {
    const body = new FormData();
    body.append('file', file);
    body.append('mode', mode);
    const response = await ensureOk(
      await fetch('/api/assessments', { method: 'POST', body }),
    );
    return response.json();
  },

  async createSample(mode: AssessmentMode): Promise<Assessment> {
    const body = new FormData();
    body.append('mode', mode);
    const response = await ensureOk(
      await fetch('/api/assessments/sample', { method: 'POST', body }),
    );
    return response.json();
  },

  async askAssessment(
    jobId: string,
    query: string,
    sessionId?: string,
  ): Promise<{ answer: string; sessionId: string; source?: string }> {
    const response = await ensureOk(
      await fetch('/api/copilot/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          query,
          session_id: sessionId,
          user_id: 'workloadiq-web-user',
        }),
      }),
    );
    const payload = await response.json();
    return {
      answer: payload.answer,
      sessionId: payload.session_id,
      source: payload.source,
    };
  },

  downloadUrl(jobId: string): string {
    return `/api/download?job_id=${encodeURIComponent(jobId)}`;
  },
};
