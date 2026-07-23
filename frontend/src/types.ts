export type AssessmentMode = 'workload' | 'quality';
export type AssessmentStatus = 'queued' | 'running' | 'completed' | 'failed';
export type WorkspaceView = 'summary' | 'drivers' | 'quality' | 'readiness';

export interface DistributionRow {
  bucket: string;
  count: number;
  pct: number;
  cum_pct: number;
}

export interface AssessmentProgress {
  message: string;
  stage?: string | null;
  percent: number;
  rows?: number | null;
  total?: number | null;
}

export interface QualityStats {
  total_tickets: number;
  evaluated_tickets: number;
  excluded_from_average: number;
  avg_score: number;
  pass_threshold?: number;
  pass_count: number;
  fail_count: number;
  fatal_count: number;
}

export interface QualityOpportunity {
  id: string;
  label: string;
  section: string;
  lost_points: number;
  issue_count: number;
}

export interface QualityTicket {
  ticket_id?: string;
  overall_score?: number;
  verdict?: string;
  fatal_found?: boolean;
  summary_feedback?: string;
  human_review_required?: string;
}

export interface AssessmentResult {
  job_id?: string;
  kind?: string;
  summary_text?: string;
  total_tickets?: number;
  top_categories?: DistributionRow[];
  top_category_subcategories?: DistributionRow[];
  top_assignment_groups?: DistributionRow[];
  detected_columns?: Record<string, string | null>;
  time_stats?: Record<string, number | boolean | string>;
  fix_notes_stats?: Record<string, number | boolean | string>;
  status_stats?: {
    has_status?: boolean;
    top_status_counts?: Array<{ status: string; count: number }>;
    distinct_status_count?: number;
    non_empty_ratio?: number;
  };
  llm_insights?: {
    parsed_json?: {
      top_heavy_hitters?: Array<{
        rank?: number;
        theme?: string;
        primary_bucket?: string;
        why_it_happens?: string;
        recommended_actions?: string[] | string;
        automation_candidates?: string[] | string;
      }>;
      cross_cutting_observations?: string[];
      data_gaps?: string[];
    };
  };
  stats?: QualityStats;
  quality_insights?: {
    section_scores?: Array<{ section: string; score_pct: number }>;
    top_opportunities?: QualityOpportunity[];
    ticket_preview?: QualityTicket[];
  };
  normalization_report?: Record<string, unknown>;
  output_csv_filename?: string;
  error?: string;
}

export interface Assessment {
  job_id: string;
  mode: AssessmentMode;
  original_name: string;
  created_at?: string;
  updated_at?: string;
  status: AssessmentStatus;
  error?: string | null;
  progress: AssessmentProgress;
  result?: AssessmentResult | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
}
