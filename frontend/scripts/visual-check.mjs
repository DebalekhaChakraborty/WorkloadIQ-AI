import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const PORT = 8092;
const BASE_URL = `http://127.0.0.1:${PORT}`;
const PYTHON = process.env.PYTHON_BIN || 'python';

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForHealth() {
  const deadline = Date.now() + 70_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${BASE_URL}/api/health`);
      if (response.ok) return;
    } catch {
      // The service is still importing.
    }
    await sleep(500);
  }
  throw new Error('Visual test service did not become healthy.');
}

async function createCompletedSample() {
  const created = await fetch(`${BASE_URL}/api/assessments/sample`, {
    method: 'POST',
    body: new URLSearchParams({ mode: 'workload' }),
  });
  if (!created.ok) throw new Error(`Sample request failed: ${created.status}`);
  const job = await created.json();

  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    const response = await fetch(`${BASE_URL}/api/assessments/${job.job_id}`);
    const assessment = await response.json();
    if (assessment.status === 'completed') return assessment;
    if (assessment.status === 'failed') throw new Error(assessment.error || 'Sample assessment failed.');
    await sleep(600);
  }
  throw new Error('Sample assessment did not complete.');
}

async function assertNoPageOverflow(page, label) {
  const dimensions = await page.evaluate(() => ({
    viewport: window.innerWidth,
    document: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }));
  if (dimensions.document > dimensions.viewport || dimensions.body > dimensions.viewport) {
    throw new Error(`${label} has horizontal page overflow: ${JSON.stringify(dimensions)}`);
  }
}

async function main() {
  const outputDirectory = await mkdtemp(path.join(tmpdir(), 'workloadiq-visual-'));
  const server = spawn(PYTHON, ['-m', 'workload_analysis.server_with_upload'], {
    cwd: ROOT,
    env: {
      ...process.env,
      GOOGLE_CLOUD_PROJECT: '',
      GOOGLE_CLOUD_LOCATION: '',
      NORMALIZE_USE_LLM: 'false',
      TICKET_QA_OUTPUT_DIR: outputDirectory,
      MPLCONFIGDIR: '/tmp/workloadiq-matplotlib',
      PORT: String(PORT),
    },
    stdio: 'ignore',
  });

  let browser;
  try {
    await waitForHealth();
    await createCompletedSample();

    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });

    const desktop = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 1 });
    await desktop.goto(BASE_URL, { waitUntil: 'networkidle' });
    await desktop.getByRole('heading', { name: 'service_desk_tickets' }).waitFor();
    await assertNoPageOverflow(desktop, 'Desktop workspace');
    await desktop.screenshot({ path: '/tmp/workloadiq-desktop.png', fullPage: true });

    await desktop.getByRole('button', { name: 'Drivers', exact: true }).click();
    await desktop.getByRole('heading', { name: 'No generated priorities returned' }).waitFor();
    const fabricatedFallbacks = await desktop
      .getByText(/Standardize triage and resolution guidance|Automated intake classification and routing/)
      .count();
    if (fabricatedFallbacks > 0) {
      throw new Error('Drivers view rendered a fabricated analytical fallback.');
    }
    await desktop.screenshot({ path: '/tmp/workloadiq-drivers-no-llm.png', fullPage: true });

    await desktop.getByRole('button', { name: 'Open assessment assistant' }).click();
    await desktop.locator('.chat-drawer.open').waitFor();
    await sleep(300);
    await desktop.screenshot({ path: '/tmp/workloadiq-chat.png', fullPage: true });

    const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
    await mobile.goto(BASE_URL, { waitUntil: 'networkidle' });
    await mobile.getByRole('heading', { name: 'service_desk_tickets' }).waitFor();
    await assertNoPageOverflow(mobile, 'Mobile workspace');
    await mobile.screenshot({ path: '/tmp/workloadiq-mobile.png', fullPage: true });

    await mobile.getByRole('button', { name: 'Open menu' }).click();
    await mobile.locator('.sidebar-mobile-open').waitFor();
    await sleep(300);
    await mobile.screenshot({ path: '/tmp/workloadiq-mobile-menu.png', fullPage: true });

    console.log('Visual checks passed: /tmp/workloadiq-desktop.png, /tmp/workloadiq-chat.png, /tmp/workloadiq-mobile.png');
  } finally {
    await browser?.close();
    server.kill('SIGTERM');
    await rm(outputDirectory, { recursive: true, force: true });
  }
}

await main();
