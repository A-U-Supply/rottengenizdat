/** Worker environment bindings (secrets set via wrangler). */
export interface Env {
  SLACK_SIGNING_SECRET: string;
  GITHUB_TOKEN: string;
  SLACK_BOT_TOKEN: string;
  SLACK_WORKSPACE: string;
}

/** Parsed result from a slash command text field. */
export interface ParsedCommand {
  command: string;
  urls: string[];
}

/** A single line item from the GitHub billing usage API. */
export interface UsageItem {
  date: string;
  product: string;
  sku: string;
  quantity: number;
  unitType: string;
  repositoryName: string;
}

/** Aggregated Actions minutes usage for the current billing cycle. */
export interface ActionsUsage {
  orgMinutes: number;
  rottengenezdatMinutes: number;
  includedMinutes: number;
  month: string;
}

/** A single GitHub Actions workflow run. */
export interface WorkflowRun {
  status: string;
  conclusion: string | null;
  created_at: string;
  updated_at: string;
  html_url: string;
  run_started_at: string;
  event: string;
}
