import type { ActionsUsage, Env, UsageItem, WorkflowRun } from "./types";

// ---------------------------------------------------------------------------
// Workflow dispatch
// ---------------------------------------------------------------------------

/**
 * Dispatch the rottengenizdat GitHub Actions workflow via the REST API.
 * Returns true on success, false on failure.
 */
export async function dispatchWorkflow(
  env: Env,
  recipe: string,
  sampleSaleCount: number = 3,
  inputMode: string = "splice",
  audioUrls: string[] = [],
): Promise<boolean> {
  const inputs: Record<string, string> = {
    recipe: recipe || "",
    sample_sale_count: String(sampleSaleCount),
    input_mode: inputMode,
    audio_urls: audioUrls.length > 0 ? audioUrls.join(",") : "",
  };

  const response = await fetch(
    "https://api.github.com/repos/A-U-Supply/rottengenizdat/actions/workflows/rottengenizdat.yml/dispatches",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "User-Agent": "rottengenizdat-slash-command/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      body: JSON.stringify({ ref: "main", inputs }),
    },
  );

  if (response.status !== 204) {
    const text = await response.text();
    console.error(`GitHub dispatch failed: ${response.status} ${text}`);
  }
  return response.status === 204;
}

/**
 * Dispatch a custom chain workflow (no recipe, raw step strings).
 */
export async function dispatchChainWorkflow(
  env: Env,
  chainSteps: string[],
  chainMode: string = "sequential",
  sampleSaleCount: number = 3,
  inputMode: string = "splice",
  audioUrls: string[] = [],
): Promise<boolean> {
  const inputs: Record<string, string> = {
    run_mode: "chain",
    chain_steps: JSON.stringify(chainSteps),
    chain_mode: chainMode,
    sample_sale_count: String(sampleSaleCount),
    input_mode: inputMode,
    audio_urls: audioUrls.length > 0 ? audioUrls.join(",") : "",
  };

  const response = await fetch(
    "https://api.github.com/repos/A-U-Supply/rottengenizdat/actions/workflows/rottengenizdat.yml/dispatches",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "User-Agent": "rottengenizdat-slash-command/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      body: JSON.stringify({ ref: "main", inputs }),
    },
  );

  if (response.status !== 204) {
    const text = await response.text();
    console.error(`GitHub chain dispatch failed: ${response.status} ${text}`);
  }
  return response.status === 204;
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/** Format a duration in seconds to a human-readable string. */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
}

/** Format a single workflow run for Slack display. */
export function formatRun(run: WorkflowRun): string {
  const statusEmoji =
    run.status === "completed"
      ? run.conclusion === "success"
        ? ":white_check_mark:"
        : ":x:"
      : run.status === "in_progress"
        ? ":hourglass_flowing_sand:"
        : ":clock1:";

  const label =
    run.status === "completed"
      ? run.conclusion === "success"
        ? "success"
        : (run.conclusion ?? "failed")
      : run.status;

  const started = new Date(run.run_started_at || run.created_at);
  const updated = new Date(run.updated_at);
  const durationSec = Math.round((updated.getTime() - started.getTime()) / 1000);
  const duration =
    run.status === "completed" ? ` in ${formatDuration(durationSec)}` : "";

  const timeAgo = Math.round((Date.now() - started.getTime()) / 60000);
  const when = timeAgo < 1 ? "just now" : `${timeAgo}m ago`;

  const trigger = run.event === "schedule" ? "scheduled" : "manual";

  return `${statusEmoji} *${label}* (${trigger}, ${when}${duration}) -- <${run.html_url}|logs>`;
}

/**
 * Fetch recent rottengenizdat workflow runs from the GitHub Actions API.
 */
export async function fetchWorkflowRuns(env: Env): Promise<WorkflowRun[]> {
  const response = await fetch(
    "https://api.github.com/repos/A-U-Supply/rottengenizdat/actions/workflows/rottengenizdat.yml/runs?per_page=3",
    {
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "User-Agent": "rottengenizdat-slash-command/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
      },
    },
  );

  if (!response.ok) {
    return [];
  }

  const data = (await response.json()) as { workflow_runs: WorkflowRun[] };
  return data.workflow_runs;
}

// ---------------------------------------------------------------------------
// Billing / usage
// ---------------------------------------------------------------------------

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

const INCLUDED_MINUTES = 2000;

/**
 * Fetch GitHub Actions usage for the current billing month.
 */
export async function fetchActionsUsage(env: Env): Promise<ActionsUsage | null> {
  try {
    const now = new Date();
    const year = now.getUTCFullYear();
    const month = now.getUTCMonth() + 1;

    const response = await fetch(
      `https://api.github.com/orgs/A-U-Supply/settings/billing/usage?year=${year}&month=${month}`,
      {
        headers: {
          Authorization: `Bearer ${env.GITHUB_TOKEN}`,
          Accept: "application/vnd.github+json",
          "User-Agent": "rottengenizdat-slash-command/1.0",
          "X-GitHub-Api-Version": "2022-11-28",
        },
      },
    );

    if (!response.ok) {
      console.error(`GitHub billing API failed: ${response.status} ${response.statusText}`);
      return null;
    }

    const data = (await response.json()) as { usageItems: UsageItem[] };
    const items = (data.usageItems ?? []).filter(
      (i) => i.product === "actions" && i.unitType === "Minutes",
    );

    let orgMinutes = 0;
    let rottengenezdatMinutes = 0;
    for (const item of items) {
      orgMinutes += item.quantity;
      if (item.repositoryName === "rottengenizdat") {
        rottengenezdatMinutes += item.quantity;
      }
    }

    return {
      orgMinutes: Math.round(orgMinutes),
      rottengenezdatMinutes: Math.round(rottengenezdatMinutes),
      includedMinutes: INCLUDED_MINUTES,
      month: MONTH_NAMES[month - 1],
    };
  } catch {
    return null;
  }
}
