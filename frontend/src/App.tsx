import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bot,
  Check,
  ChevronRight,
  CircleGauge,
  Clock3,
  Download,
  FileSearch,
  FileSpreadsheet,
  History,
  Layers3,
  Menu,
  MessageSquareText,
  PanelLeftClose,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  TableProperties,
  Upload,
  X,
  XCircle,
} from 'lucide-react';
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';
import type {
  Assessment,
  AssessmentMode,
  AssessmentResult,
  ChatMessage,
  DistributionRow,
  WorkspaceView,
} from './types';

const VIEW_ITEMS: Array<{
  id: WorkspaceView;
  label: string;
  icon: typeof BarChart3;
}> = [
  { id: 'summary', label: 'Summary', icon: CircleGauge },
  { id: 'drivers', label: 'Drivers', icon: BarChart3 },
  { id: 'quality', label: 'Quality', icon: ShieldCheck },
  { id: 'readiness', label: 'Data readiness', icon: TableProperties },
];

const QUALITY_VIEW_ITEMS: typeof VIEW_ITEMS = [
  { id: 'summary', label: 'Summary', icon: CircleGauge },
  { id: 'drivers', label: 'Controls', icon: ShieldCheck },
  { id: 'quality', label: 'Review queue', icon: FileSearch },
  { id: 'readiness', label: 'Data readiness', icon: TableProperties },
];

const formatDate = (value?: string) => {
  if (!value) return 'Just now';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Just now';
  return new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};

const formatPercent = (value?: number, digits = 0) =>
  `${((value || 0) * 100).toFixed(digits)}%`;

const titleCase = (value: string) =>
  value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const asList = (value?: string[] | string): string[] => {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
};

function Brand() {
  return (
    <div className="brand">
      <span className="brand-mark" aria-hidden="true">
        <Activity size={19} strokeWidth={2.4} />
      </span>
      <span>
        <strong>WorkloadIQ</strong>
        <small>Assessment Studio</small>
      </span>
    </div>
  );
}

function StatusBadge({ status }: { status: Assessment['status'] }) {
  const labels = {
    queued: 'Queued',
    running: 'Analyzing',
    completed: 'Complete',
    failed: 'Failed',
  };
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-dot" />
      {labels[status]}
    </span>
  );
}

function Sidebar({
  jobs,
  selectedId,
  selectedMode,
  view,
  mobileOpen,
  onCloseMobile,
  onSelect,
  onChangeView,
  onNew,
}: {
  jobs: Assessment[];
  selectedId: string | null;
  selectedMode?: AssessmentMode;
  view: WorkspaceView;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  onSelect: (jobId: string) => void;
  onChangeView: (view: WorkspaceView) => void;
  onNew: () => void;
}) {
  const navigationItems = selectedMode === 'quality' ? QUALITY_VIEW_ITEMS : VIEW_ITEMS;
  return (
    <>
      {mobileOpen && <button className="mobile-scrim" onClick={onCloseMobile} aria-label="Close menu" />}
      <aside className={`sidebar ${mobileOpen ? 'sidebar-mobile-open' : ''}`}>
        <div className="sidebar-head">
          <Brand />
          <button className="icon-button mobile-only" onClick={onCloseMobile} aria-label="Close menu">
            <PanelLeftClose size={18} />
          </button>
        </div>

        <button className="new-assessment-button" onClick={onNew}>
          <Plus size={17} />
          New assessment
        </button>

        <nav className="primary-nav" aria-label="Assessment views">
          <p className="nav-label">Workspace</p>
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={view === item.id ? 'nav-item active' : 'nav-item'}
                onClick={() => {
                  onChangeView(item.id);
                  onCloseMobile();
                }}
              >
                <Icon size={17} />
                {item.label}
              </button>
            );
          })}
        </nav>

        <section className="recent-section">
          <div className="recent-heading">
            <p className="nav-label">Recent</p>
            <History size={14} />
          </div>
          <div className="recent-list">
            {jobs.length === 0 && <p className="recent-empty">No assessments yet</p>}
            {jobs.slice(0, 8).map((job) => (
              <button
                key={job.job_id}
                className={selectedId === job.job_id ? 'recent-item active' : 'recent-item'}
                onClick={() => {
                  onSelect(job.job_id);
                  onCloseMobile();
                }}
              >
                <span className="recent-icon">
                  {job.mode === 'quality' ? <ShieldCheck size={15} /> : <FileSearch size={15} />}
                </span>
                <span className="recent-copy">
                  <strong>{job.original_name}</strong>
                  <small>{formatDate(job.created_at)}</small>
                </span>
                <span className={`mini-status mini-${job.status}`} title={job.status} />
              </button>
            ))}
          </div>
        </section>

        <div className="sidebar-foot">
          <div className="workspace-avatar">DC</div>
          <div>
            <strong>Assessment workspace</strong>
            <small>Service operations</small>
          </div>
        </div>
      </aside>
    </>
  );
}

function UploadDialog({
  open,
  busy,
  onClose,
  onSubmit,
  onSample,
}: {
  open: boolean;
  busy: boolean;
  onClose: () => void;
  onSubmit: (file: File, mode: AssessmentMode) => Promise<void>;
  onSample: (mode: AssessmentMode) => Promise<void>;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<AssessmentMode>('workload');
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    if (!open) {
      setFile(null);
      setDragActive(false);
    }
  }, [open]);

  if (!open) return null;

  const acceptFile = (candidate?: File) => {
    if (candidate?.name.toLowerCase().endsWith('.csv')) setFile(candidate);
  };

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="upload-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="dialog-header">
          <div>
            <p className="eyebrow">New assessment</p>
            <h2 id="upload-title">Analyze a ticket dataset</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close">
            <X size={19} />
          </button>
        </div>

        <div className="mode-control" role="radiogroup" aria-label="Assessment mode">
          <button
            className={mode === 'workload' ? 'active' : ''}
            onClick={() => setMode('workload')}
            role="radio"
            aria-checked={mode === 'workload'}
          >
            <BarChart3 size={17} />
            <span>
              <strong>Workload intelligence</strong>
              <small>Volume, drivers, queues and trends</small>
            </span>
          </button>
          <button
            className={mode === 'quality' ? 'active' : ''}
            onClick={() => setMode('quality')}
            role="radio"
            aria-checked={mode === 'quality'}
          >
            <ShieldCheck size={17} />
            <span>
              <strong>Ticket quality audit</strong>
              <small>Scoring, controls and exceptions</small>
            </span>
          </button>
        </div>

        <button
          className={`drop-zone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragActive(false);
            acceptFile(event.dataTransfer.files[0]);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => acceptFile(event.target.files?.[0])}
            hidden
          />
          <span className="drop-icon">
            {file ? <FileSpreadsheet size={25} /> : <Upload size={25} />}
          </span>
          {file ? (
            <>
              <strong>{file.name}</strong>
              <small>{(file.size / 1024).toFixed(1)} KB</small>
            </>
          ) : (
            <>
              <strong>Drop a CSV here</strong>
              <small>or select from your computer · 25 MB max</small>
            </>
          )}
        </button>

        <div className="dialog-actions">
          <button className="text-button" onClick={() => onSample(mode)} disabled={busy}>
            <Sparkles size={16} />
            Use sample data
          </button>
          <button
            className="primary-button"
            onClick={() => file && onSubmit(file, mode)}
            disabled={!file || busy}
          >
            {busy ? <RefreshCw className="spin" size={17} /> : <ArrowUpRight size={17} />}
            {busy ? 'Starting...' : 'Run assessment'}
          </button>
        </div>
      </section>
    </div>
  );
}

function EmptyWorkspace({ onNew, onSample }: { onNew: () => void; onSample: () => void }) {
  return (
    <div className="empty-workspace">
      <header className="page-heading">
        <p className="eyebrow">Assessment workspace</p>
        <h1>Service desk intelligence</h1>
        <p>Operational evidence for workload planning and ticket quality.</p>
      </header>

      <section className="start-band">
        <div className="start-visual" aria-hidden="true">
          <div className="signal-grid">
            {[64, 42, 78, 55, 90, 72, 46, 82].map((height, index) => (
              <span key={index} style={{ height: `${height}%` }} />
            ))}
          </div>
          <div className="signal-caption">
            <span>Ticket volume</span>
            <strong>Ready for analysis</strong>
          </div>
        </div>
        <div className="start-copy">
          <span className="feature-icon">
            <Layers3 size={22} />
          </span>
          <h2>Start with a ticket export</h2>
          <p>CSV · ServiceNow, Jira and common service desk schemas</p>
          <div className="start-actions">
            <button className="primary-button" onClick={onNew}>
              <Upload size={17} />
              Select dataset
            </button>
            <button className="secondary-button" onClick={onSample}>
              <Sparkles size={16} />
              Sample assessment
            </button>
          </div>
        </div>
      </section>

      <section className="assessment-profiles">
        <article>
          <BarChart3 size={20} />
          <div>
            <strong>Workload intelligence</strong>
            <p>Demand concentration, assignment load, resolution signals and action priorities.</p>
          </div>
          <ChevronRight size={17} />
        </article>
        <article>
          <ShieldCheck size={20} />
          <div>
            <strong>Ticket quality audit</strong>
            <p>Control scoring, fatal exceptions, documentation gaps and review queues.</p>
          </div>
          <ChevronRight size={17} />
        </article>
      </section>
    </div>
  );
}

function ProgressWorkspace({ assessment }: { assessment: Assessment }) {
  const percent = assessment.progress.percent || 0;
  return (
    <div className="progress-workspace">
      <div className="progress-orbit">
        <svg viewBox="0 0 120 120" aria-hidden="true">
          <circle cx="60" cy="60" r="52" className="orbit-track" />
          <circle
            cx="60"
            cy="60"
            r="52"
            className="orbit-value"
            pathLength="100"
            strokeDasharray={`${percent} 100`}
          />
        </svg>
        <span>{percent}%</span>
      </div>
      <p className="eyebrow">{assessment.mode === 'quality' ? 'Quality audit' : 'Workload intelligence'}</p>
      <h2>{assessment.progress.message}</h2>
      <p className="progress-file">
        <FileSpreadsheet size={16} />
        {assessment.original_name}
      </p>
      <div className="stage-rail">
        {['Ingest', 'Normalize', 'Analyze', 'Synthesize'].map((stage, index) => (
          <span key={stage} className={percent >= [10, 30, 55, 80][index] ? 'complete' : ''}>
            <Check size={13} />
            {stage}
          </span>
        ))}
      </div>
    </div>
  );
}

function Kpi({
  label,
  value,
  detail,
  tone = 'neutral',
  icon: Icon,
}: {
  label: string;
  value: string;
  detail: string;
  tone?: 'neutral' | 'good' | 'warn' | 'blue';
  icon: typeof Activity;
}) {
  return (
    <article className={`kpi kpi-${tone}`}>
      <div className="kpi-head">
        <span>{label}</span>
        <Icon size={17} />
      </div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function DistributionChart({
  rows,
  title,
  subtitle,
}: {
  rows: DistributionRow[];
  title: string;
  subtitle: string;
}) {
  const max = Math.max(...rows.map((row) => row.count), 1);
  return (
    <section className="panel distribution-panel">
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <span className="panel-meta">Top {Math.min(rows.length, 7)}</span>
      </div>
      <div className="distribution-chart">
        {rows.slice(0, 7).map((row, index) => (
          <div className="distribution-row" key={`${row.bucket}-${index}`}>
            <div className="distribution-label">
              <span>{row.bucket}</span>
              <strong>{row.count}</strong>
            </div>
            <div className="bar-track">
              <span
                className={index === 0 ? 'bar-value primary' : 'bar-value'}
                style={{ width: `${Math.max(3, (row.count / max) * 100)}%` }}
              />
            </div>
            <span className="distribution-pct">{formatPercent(row.pct, 1)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function AiBrief({ result }: { result: AssessmentResult }) {
  const parsed = result.llm_insights?.parsed_json;
  const lead = parsed?.top_heavy_hitters?.[0];
  const observations = parsed?.cross_cutting_observations || [];
  const categories = result.top_categories || [];
  const deterministic = categories[0]
    ? `${categories[0].bucket} accounts for ${formatPercent(categories[0].pct, 1)} of observed demand.`
    : 'No dominant category was detected in the available fields.';

  return (
    <section className="panel ai-brief">
      <div className="panel-heading">
        <div>
          <h3>Assessment brief</h3>
          <p>Grounded synthesis</p>
        </div>
        <span className="ai-signal">
          <Sparkles size={14} />
          AI
        </span>
      </div>
      <div className="brief-lead">
        <span className="brief-rank">{lead?.rank || 1}</span>
        <div>
          <small>Primary opportunity</small>
          <strong>{lead?.theme || categories[0]?.bucket || 'Demand concentration'}</strong>
          <p>{lead?.why_it_happens || deterministic}</p>
        </div>
      </div>
      <div className="brief-observations">
        {(observations.length ? observations : [deterministic]).slice(0, 3).map((observation, index) => (
          <p key={index}>
            <Check size={14} />
            <span>{observation}</span>
          </p>
        ))}
      </div>
    </section>
  );
}

function StatusMix({ result }: { result: AssessmentResult }) {
  const rows = result.status_stats?.top_status_counts || [];
  const total = rows.reduce((sum, row) => sum + row.count, 0) || 1;
  const colors = ['status-green', 'status-blue', 'status-amber', 'status-coral', 'status-gray'];
  return (
    <section className="panel status-panel">
      <div className="panel-heading">
        <div>
          <h3>Status mix</h3>
          <p>{result.status_stats?.distinct_status_count || rows.length} workflow states</p>
        </div>
      </div>
      <div className="status-strip" aria-label="Ticket status distribution">
        {rows.slice(0, 5).map((row, index) => (
          <span
            key={row.status}
            className={colors[index]}
            style={{ width: `${(row.count / total) * 100}%` }}
            title={`${row.status}: ${row.count}`}
          />
        ))}
      </div>
      <div className="status-legend">
        {rows.slice(0, 5).map((row, index) => (
          <div key={row.status}>
            <span className={colors[index]} />
            <p>
              <strong>{row.status}</strong>
              <small>{row.count} tickets</small>
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

function WorkloadSummary({ result }: { result: AssessmentResult }) {
  const categories = result.top_categories || [];
  const groups = result.top_assignment_groups || [];
  const total = result.total_tickets || 0;
  const fixCoverageValue = result.fix_notes_stats?.fix_notes_non_empty_ratio;
  const fixCoverage = typeof fixCoverageValue === 'number' ? fixCoverageValue : null;
  const fixNoteRows = result.fix_notes_stats?.fix_notes_non_empty_rows;
  const medianHoursValue = result.time_stats?.resolution_time_hours_median;
  const medianHours = typeof medianHoursValue === 'number' ? medianHoursValue : null;
  const resolvedRows = result.time_stats?.resolved_rows;
  const topShare = categories[0]?.pct;

  return (
    <>
      <div className="kpi-grid">
        <Kpi label="Tickets analyzed" value={total.toLocaleString()} detail="Normalized records" icon={FileSearch} />
        <Kpi
          label="Largest driver"
          value={topShare == null ? 'N/A' : formatPercent(topShare, 1)}
          detail={categories[0]?.bucket || 'Not detected'}
          tone="warn"
          icon={ArrowUpRight}
        />
        <Kpi
          label="Fix-note coverage"
          value={fixCoverage == null ? 'N/A' : formatPercent(fixCoverage)}
          detail={typeof fixNoteRows === 'number' ? `${fixNoteRows} documented tickets` : 'Coverage not available'}
          tone={fixCoverage == null ? 'neutral' : fixCoverage >= 0.8 ? 'good' : 'warn'}
          icon={ShieldCheck}
        />
        <Kpi
          label="Median resolution"
          value={medianHours == null ? 'N/A' : `${medianHours.toFixed(1)}h`}
          detail={typeof resolvedRows === 'number' ? `${resolvedRows} resolved records` : 'Resolution data not available'}
          tone="blue"
          icon={Clock3}
        />
      </div>
      <div className="analysis-grid">
        <DistributionChart rows={categories} title="Demand concentration" subtitle="Category share of ticket volume" />
        <AiBrief result={result} />
      </div>
      <div className="analysis-grid lower-grid">
        <DistributionChart rows={groups} title="Assignment load" subtitle="Tickets handled by support group" />
        <StatusMix result={result} />
      </div>
    </>
  );
}

function DriverView({ result }: { result: AssessmentResult }) {
  const hitters = result.llm_insights?.parsed_json?.top_heavy_hitters || [];
  const combinations = result.top_category_subcategories || [];
  return (
    <div className="drivers-layout">
      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h3>Category and subcategory</h3>
            <p>Most frequent demand combinations</p>
          </div>
        </div>
        <div className="data-table">
          <div className="table-row table-head">
            <span>Driver</span>
            <span>Tickets</span>
            <span>Share</span>
            <span>Cumulative</span>
          </div>
          {combinations.slice(0, 10).map((row, index) => (
            <div className="table-row" key={`${row.bucket}-${index}`}>
              <span>
                <i>{String(index + 1).padStart(2, '0')}</i>
                {row.bucket}
              </span>
              <strong>{row.count}</strong>
              <span>{formatPercent(row.pct, 1)}</span>
              <span>{formatPercent(row.cum_pct, 1)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="priority-list">
        <div className="section-heading">
          <div>
            <h3>Response priorities</h3>
            <p>Ranked opportunities from the assessment</p>
          </div>
        </div>
        {hitters.length === 0 ? (
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h3>No generated priorities returned</h3>
                <p>The measured driver table remains available above.</p>
              </div>
            </div>
            <p className="muted-copy">
              This assessment did not return a valid model-generated priority block, so no actions or automation
              recommendations are being inferred.
            </p>
          </section>
        ) : (
          hitters.slice(0, 5).map((hitter, index) => (
            <article className="priority-item" key={`${hitter.theme}-${index}`}>
              <span className="priority-rank">{hitter.rank || index + 1}</span>
              <div className="priority-main">
                <div className="priority-title">
                  <h4>{hitter.theme || hitter.primary_bucket || 'Untitled assessment finding'}</h4>
                  {hitter.primary_bucket && <span>{hitter.primary_bucket}</span>}
                </div>
                <p>{hitter.why_it_happens || 'No causal explanation was returned for this priority.'}</p>
                <div className="action-columns">
                  <div>
                    <small>Recommended action</small>
                    <strong>{asList(hitter.recommended_actions)[0] || 'Not provided by this assessment.'}</strong>
                  </div>
                  <div>
                    <small>Automation candidate</small>
                    <strong>{asList(hitter.automation_candidates)[0] || 'Not provided by this assessment.'}</strong>
                  </div>
                </div>
              </div>
            </article>
          ))
        )}
      </section>
    </div>
  );
}

function QualitySummary({ result }: { result: AssessmentResult }) {
  const stats = result.stats;
  const quality = result.quality_insights;
  if (!stats) return null;
  const passRate = stats.evaluated_tickets ? stats.pass_count / stats.evaluated_tickets : 0;
  const meetsThreshold =
    typeof stats.pass_threshold === 'number' ? stats.avg_score >= stats.pass_threshold : null;
  const sectionRows = quality?.section_scores || [];
  const chartRows: DistributionRow[] = sectionRows.map((row, index) => ({
    bucket: titleCase(row.section),
    count: row.score_pct,
    pct: row.score_pct / 100,
    cum_pct: index === 0 ? row.score_pct / 100 : 0,
  }));
  return (
    <>
      <div className="kpi-grid">
        <Kpi label="Average quality" value={`${stats.avg_score.toFixed(1)}`} detail="Score out of 100" tone="blue" icon={CircleGauge} />
        <Kpi label="Pass rate" value={formatPercent(passRate, 1)} detail={`${stats.pass_count} passing tickets`} tone={stats.fail_count === 0 ? 'good' : 'warn'} icon={ShieldCheck} />
        <Kpi label="Failed controls" value={stats.fail_count.toString()} detail="Tickets below threshold" tone="warn" icon={XCircle} />
        <Kpi label="Fatal exceptions" value={stats.fatal_count.toString()} detail={`${stats.excluded_from_average} manual reviews`} icon={AlertTriangle} />
      </div>
      <div className="analysis-grid">
        <DistributionChart rows={chartRows} title="Control performance" subtitle="Average score by quality section" />
        <section className="panel qa-brief">
          <div className="panel-heading">
            <div>
              <h3>Audit position</h3>
              <p>{stats.evaluated_tickets} scored tickets</p>
            </div>
            <span className={meetsThreshold === false ? 'score-ring warn' : meetsThreshold ? 'score-ring good' : 'score-ring'}>
              {Math.round(stats.avg_score)}
            </span>
          </div>
          <div className="qa-split">
            <div>
              <span className="qa-number pass">{stats.pass_count}</span>
              <small>Pass</small>
            </div>
            <div>
              <span className="qa-number fail">{stats.fail_count}</span>
              <small>Fail</small>
            </div>
            <div>
              <span className="qa-number review">{stats.excluded_from_average}</span>
              <small>Review</small>
            </div>
          </div>
          <p className="qa-note">
            {meetsThreshold == null
              ? 'Pass and fail verdicts reflect the threshold recorded by the assessment engine.'
              : `The assessed population is ${meetsThreshold ? 'above' : 'below'} the configured ${stats.pass_threshold}/100 quality threshold.`}
          </p>
        </section>
      </div>
      <QualityControls result={result} compact />
    </>
  );
}

function QualityControls({ result, compact = false }: { result: AssessmentResult; compact?: boolean }) {
  const opportunities = result.quality_insights?.top_opportunities || [];
  const tickets = result.quality_insights?.ticket_preview || [];
  return (
    <div className={compact ? 'quality-panels compact' : 'quality-panels'}>
      <section className="panel opportunities-panel">
        <div className="panel-heading">
          <div>
            <h3>Largest control gaps</h3>
            <p>Ranked by points lost across the audit</p>
          </div>
        </div>
        <div className="opportunity-list">
          {opportunities.slice(0, compact ? 5 : 8).map((item, index) => (
            <div className="opportunity-row" key={item.id}>
              <span className="opportunity-index">{index + 1}</span>
              <div>
                <strong>{item.label}</strong>
                <small>{titleCase(item.section)}</small>
              </div>
              <span>{item.issue_count} flags</span>
              <strong>{item.lost_points} pts</strong>
            </div>
          ))}
        </div>
      </section>
      {!compact && (
        <section className="panel table-panel ticket-table">
          <div className="panel-heading">
            <div>
              <h3>Ticket review queue</h3>
              <p>Scored record preview</p>
            </div>
          </div>
          <div className="data-table">
            <div className="table-row table-head">
              <span>Ticket</span>
              <span>Score</span>
              <span>Verdict</span>
              <span>Exception</span>
            </div>
            {tickets.slice(0, 20).map((ticket, index) => (
              <div className="table-row" key={`${ticket.ticket_id}-${index}`}>
                <strong>{ticket.ticket_id || `Row ${index + 1}`}</strong>
                <span>{ticket.overall_score ?? '—'}</span>
                <span className={`verdict verdict-${String(ticket.verdict).toLowerCase()}`}>
                  {ticket.verdict || 'Review'}
                </span>
                <span>{ticket.fatal_found ? 'Fatal' : ticket.human_review_required === 'YES' ? 'Manual' : 'None'}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function WorkloadQuality({ result }: { result: AssessmentResult }) {
  const fixCoverage = Number(result.fix_notes_stats?.fix_notes_non_empty_ratio || 0);
  const statusCoverage = Number(result.status_stats?.non_empty_ratio || 0);
  const dataGaps = result.llm_insights?.parsed_json?.data_gaps || [];
  const observations = result.llm_insights?.parsed_json?.cross_cutting_observations || [];
  return (
    <div className="quality-workload">
      <section className="panel coverage-panel">
        <div className="panel-heading">
          <div>
            <h3>Evidence coverage</h3>
            <p>Completeness signals used by the assessment</p>
          </div>
        </div>
        {[
          ['Fix and work notes', fixCoverage],
          ['Workflow status', statusCoverage],
          ['Created timestamps', result.time_stats?.has_created ? 1 : 0],
          ['Closed timestamps', result.time_stats?.has_closed ? 1 : 0],
        ].map(([label, value]) => (
          <div className="coverage-row" key={String(label)}>
            <span>{label}</span>
            <div className="coverage-track">
              <span style={{ width: `${Number(value) * 100}%` }} />
            </div>
            <strong>{formatPercent(Number(value))}</strong>
          </div>
        ))}
      </section>
      <section className="panel findings-panel">
        <div className="panel-heading">
          <div>
            <h3>Data findings</h3>
            <p>Constraints and cross-cutting observations</p>
          </div>
        </div>
        <div className="finding-list">
          {[...dataGaps, ...observations].slice(0, 7).map((item, index) => (
            <div key={index}>
              {index < dataGaps.length ? <AlertTriangle size={16} /> : <Check size={16} />}
              <p>{item}</p>
            </div>
          ))}
          {dataGaps.length === 0 && observations.length === 0 && (
            <div>
              <Check size={16} />
              <p>The deterministic analysis completed without an additional model-generated data warning.</p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function ReadinessView({ result }: { result: AssessmentResult }) {
  const columns = Object.entries(result.detected_columns || {}).filter(([, value]) => value);
  const report = result.normalization_report || {};
  const reportRows = Object.entries(report)
    .filter(([, value]) => ['string', 'number', 'boolean'].includes(typeof value))
    .slice(0, 12);
  return (
    <div className="readiness-layout">
      <section className="panel mapping-panel">
        <div className="panel-heading">
          <div>
            <h3>Detected schema</h3>
            <p>{columns.length} canonical fields mapped</p>
          </div>
          <span className="readiness-score">
            {columns.length > 0 ? <Check size={14} /> : <AlertTriangle size={14} />}
            {columns.length > 0 ? 'Mapped' : 'Limited'}
          </span>
        </div>
        <div className="mapping-grid">
          {columns.map(([canonical, source]) => (
            <div key={canonical}>
              <span>{titleCase(canonical)}</span>
              <ChevronRight size={14} />
              <strong>{String(source)}</strong>
            </div>
          ))}
          {columns.length === 0 && (
            <p className="muted-copy">Canonical mapping details are available in the exported assessment.</p>
          )}
        </div>
      </section>
      <section className="panel normalization-panel">
        <div className="panel-heading">
          <div>
            <h3>Normalization report</h3>
            <p>Input preparation summary</p>
          </div>
        </div>
        <div className="report-grid">
          {reportRows.map(([key, value]) => (
            <div key={key}>
              <span>{titleCase(key)}</span>
              <strong>{String(value)}</strong>
            </div>
          ))}
          {reportRows.length === 0 && (
            <div>
              <span>Status</span>
              <strong>No scalar summary returned</strong>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function AnalysisContent({ assessment, view }: { assessment: Assessment; view: WorkspaceView }) {
  const result = assessment.result;
  if (!result) return null;
  const isQuality = assessment.mode === 'quality' || result.kind === 'qa_batch';

  if (view === 'readiness') return <ReadinessView result={result} />;
  if (isQuality) {
    if (view === 'summary') return <QualitySummary result={result} />;
    if (view === 'drivers') return <QualityControls result={result} compact />;
    return <QualityControls result={result} />;
  }
  if (view === 'summary') return <WorkloadSummary result={result} />;
  if (view === 'drivers') return <DriverView result={result} />;
  if (view === 'quality') return <WorkloadQuality result={result} />;
  return <ReadinessView result={result} />;
}

function ChatWidget({ assessment }: { assessment: Assessment | null }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const available = assessment?.status === 'completed' && Boolean(assessment.result);

  useEffect(() => {
    setMessages([]);
    setSessionId(undefined);
    setOpen(false);
  }, [assessment?.job_id]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, sending]);

  const send = async (query: string) => {
    if (!assessment || !query.trim() || sending || !available) return;
    const userMessage: ChatMessage = { id: `${Date.now()}-u`, role: 'user', content: query.trim() };
    setMessages((current) => [...current, userMessage]);
    setInput('');
    setSending(true);
    try {
      const response = await api.askAssessment(assessment.job_id, query.trim(), sessionId);
      setSessionId(response.sessionId);
      setMessages((current) => [
        ...current,
        { id: `${Date.now()}-a`, role: 'assistant', content: response.answer },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { id: `${Date.now()}-e`, role: 'system', content: error instanceof Error ? error.message : 'Assistant unavailable.' },
      ]);
    } finally {
      setSending(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void send(input);
  };

  return (
    <>
      <button
        className={`chat-launcher ${open ? 'open' : ''}`}
        onClick={() => available && setOpen((current) => !current)}
        aria-label={open ? 'Close assessment assistant' : 'Open assessment assistant'}
        title={available ? 'Ask this assessment' : 'Complete an assessment first'}
        disabled={!available}
      >
        {open ? <X size={22} /> : <MessageSquareText size={22} />}
      </button>
      <aside className={`chat-drawer ${open ? 'open' : ''}`} aria-hidden={!open}>
        <header>
          <div className="chat-avatar">
            <Bot size={19} />
          </div>
          <div>
            <strong>Assessment analyst</strong>
            <small>
              <span />
              Grounded in {assessment?.original_name}
            </small>
          </div>
          <button className="icon-button" onClick={() => setOpen(false)} aria-label="Close assistant">
            <X size={18} />
          </button>
        </header>
        <div className="chat-messages" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="chat-intro">
              <span>
                <Sparkles size={19} />
              </span>
              <strong>Ask about this assessment</strong>
              <p>Answers stay grounded in the active result.</p>
              <div className="suggestion-list">
                {['Summarize the main risk', 'What should we fix first?', 'Where are the data gaps?'].map((query) => (
                  <button key={query} onClick={() => void send(query)}>
                    {query}
                    <ChevronRight size={14} />
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((message) => (
            <div key={message.id} className={`chat-message ${message.role}`}>
              {message.role === 'assistant' && <Bot size={15} />}
              <p>{message.content}</p>
            </div>
          ))}
          {sending && (
            <div className="chat-message assistant typing">
              <Bot size={15} />
              <p>
                <i />
                <i />
                <i />
              </p>
            </div>
          )}
        </div>
        <form className="chat-composer" onSubmit={submit}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about this assessment"
            aria-label="Message"
          />
          <button type="submit" aria-label="Send" disabled={!input.trim() || sending}>
            <Send size={17} />
          </button>
        </form>
      </aside>
    </>
  );
}

export default function App() {
  const [jobs, setJobs] = useState<Assessment[]>([]);
  const [selected, setSelected] = useState<Assessment | null>(null);
  const [view, setView] = useState<WorkspaceView>('summary');
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadJob = async (jobId: string) => {
    try {
      const job = await api.getAssessment(jobId);
      setSelected(job);
      setJobs((current) => current.map((item) => (item.job_id === job.job_id ? { ...item, ...job, result: null } : item)));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Unable to load assessment.');
    }
  };

  useEffect(() => {
    const initialize = async () => {
      try {
        const recent = await api.listAssessments();
        setJobs(recent);
        if (recent[0]) await loadJob(recent[0].job_id);
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'Unable to connect to the assessment service.');
      } finally {
        setLoading(false);
      }
    };
    void initialize();
  }, []);

  useEffect(() => {
    if (!selected || !['queued', 'running'].includes(selected.status)) return;
    const timer = window.setInterval(() => void loadJob(selected.job_id), 1400);
    return () => window.clearInterval(timer);
  }, [selected?.job_id, selected?.status]);

  const beginAssessment = async (runner: () => Promise<Assessment>) => {
    setUploadBusy(true);
    setError(null);
    try {
      const job = await runner();
      setSelected(job);
      setJobs((current) => [job, ...current.filter((item) => item.job_id !== job.job_id)]);
      setView('summary');
      setUploadOpen(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Unable to start assessment.');
    } finally {
      setUploadBusy(false);
    }
  };

  const pageLabel = useMemo(() => {
    const items = selected?.mode === 'quality' ? QUALITY_VIEW_ITEMS : VIEW_ITEMS;
    return items.find((item) => item.id === view)?.label || 'Summary';
  }, [selected?.mode, view]);
  const complete = selected?.status === 'completed' && selected.result;

  return (
    <div className="app-shell">
      <Sidebar
        jobs={jobs}
        selectedId={selected?.job_id || null}
        selectedMode={selected?.mode}
        view={view}
        mobileOpen={mobileMenu}
        onCloseMobile={() => setMobileMenu(false)}
        onSelect={(jobId) => {
          setView('summary');
          void loadJob(jobId);
        }}
        onChangeView={setView}
        onNew={() => setUploadOpen(true)}
      />

      <main className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-only" onClick={() => setMobileMenu(true)} aria-label="Open menu">
              <Menu size={20} />
            </button>
            <div className="mobile-only">
              <Brand />
            </div>
            {selected && (
              <div className="breadcrumb desktop-only">
                <span>Assessments</span>
                <ChevronRight size={14} />
                <strong>{pageLabel}</strong>
              </div>
            )}
          </div>
          <div className="topbar-actions">
            <button className="icon-button desktop-only" title="Search assessments" aria-label="Search assessments">
              <Search size={18} />
            </button>
            {complete && (
              <a className="secondary-button export-button" href={api.downloadUrl(selected.job_id)}>
                <Download size={16} />
                Export
              </a>
            )}
            <button className="primary-button compact-button" onClick={() => setUploadOpen(true)}>
              <Plus size={17} />
              <span className="desktop-only">New assessment</span>
            </button>
          </div>
        </header>

        <div className="workspace-scroll">
          {loading ? (
            <div className="loading-screen">
              <RefreshCw className="spin" size={25} />
              <p>Loading workspace</p>
            </div>
          ) : !selected ? (
            <EmptyWorkspace
              onNew={() => setUploadOpen(true)}
              onSample={() => void beginAssessment(() => api.createSample('workload'))}
            />
          ) : (
            <div className="assessment-workspace">
              <header className="assessment-header">
                <div>
                  <div className="assessment-title-row">
                    <p className="eyebrow">{selected.mode === 'quality' ? 'Ticket quality audit' : 'Workload intelligence'}</p>
                    <StatusBadge status={selected.status} />
                  </div>
                  <h1>{selected.original_name.replace(/\.csv$/i, '')}</h1>
                  <p>
                    <FileSpreadsheet size={15} />
                    {selected.original_name}
                    <span>·</span>
                    {formatDate(selected.created_at)}
                  </p>
                </div>
                {selected.status === 'completed' && (
                  <div className="header-verdict">
                    <span>{selected.mode === 'quality' ? 'Audit complete' : 'Assessment complete'}</span>
                    <strong>
                      {selected.mode === 'quality'
                        ? `${selected.result?.stats?.avg_score || 0}/100`
                        : `${selected.result?.total_tickets || 0} tickets`}
                    </strong>
                  </div>
                )}
              </header>

              {selected.status === 'failed' ? (
                <section className="failure-state">
                  <AlertTriangle size={28} />
                  <h2>Assessment could not be completed</h2>
                  <p>{selected.error || selected.result?.error || 'The analysis service returned an error.'}</p>
                  <button className="primary-button" onClick={() => setUploadOpen(true)}>
                    <RefreshCw size={16} />
                    Start again
                  </button>
                </section>
              ) : selected.status !== 'completed' ? (
                <ProgressWorkspace assessment={selected} />
              ) : (
                <AnalysisContent assessment={selected} view={view} />
              )}
            </div>
          )}
        </div>
      </main>

      {error && (
        <div className="toast" role="alert">
          <AlertTriangle size={17} />
          <span>{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss">
            <X size={16} />
          </button>
        </div>
      )}

      <UploadDialog
        open={uploadOpen}
        busy={uploadBusy}
        onClose={() => !uploadBusy && setUploadOpen(false)}
        onSubmit={(file, mode) => beginAssessment(() => api.createAssessment(file, mode))}
        onSample={(mode) => beginAssessment(() => api.createSample(mode))}
      />
      <ChatWidget assessment={selected} />
    </div>
  );
}
