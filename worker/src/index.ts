import {
  isValidRecipe,
  formatRecipeList,
  suggestRecipes,
  RECIPES,
} from "./recipes";
import type { Env } from "./types";
import { verifySlackSignature, parseSlashCommand, slackResponse } from "./slack";
import { dispatchWorkflow, fetchWorkflowRuns, fetchActionsUsage } from "./github";
import { handleInteraction } from "./interactions";
import { buildStatusBlocks, buildUsageContextShort } from "./blocks";
import { buildModalView, buildHelpView } from "./modal";

// Re-export for tests
export type { Env, ParsedCommand } from "./types";
export { parseSlashCommand } from "./slack";

// ---------------------------------------------------------------------------
// Help text
// ---------------------------------------------------------------------------

function buildHelpText(): string {
  return [
    "*rottengenizdat* -- bone music for the machine age :radio:",
    "",
    "*Usage:*",
    "  `/rottengenizdat` -- open the recipe picker modal",
    "  `/rottengenizdat <recipe>` -- run a specific recipe with 3 random samples",
    "  `/rottengenizdat list` -- show all available recipes",
    "  `/rottengenizdat status` -- check recent run status",
    "  `/rottengenizdat help` -- show this message",
    "",
    `*Recipes:*`,
    `  ${RECIPES.length} recipes available. Use \`/rottengenizdat list\` to see them all.`,
    "",
    "*How it works:*",
    "  1. Your command triggers a processing pipeline",
    "  2. Random audio samples are pulled from #sample-sale",
    "  3. The recipe feeds them through RAVE neural networks",
    "  4. The result is posted to the channel",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Command routing
// ---------------------------------------------------------------------------

export async function handleSlashCommand(body: string, env: Env, ctx: ExecutionContext): Promise<Response> {
  const params = new URLSearchParams(body);
  const rawText = (params.get("text") ?? "").trim();
  const { command, urls } = parseSlashCommand(rawText);

  // Help (modal)
  if (command === "help") {
    const triggerId = params.get("trigger_id");
    if (triggerId) {
      const error = await openViewModal(env, triggerId, buildHelpView());
      if (error) {
        return slackResponse(`:x: Failed to open help: ${error}`);
      }
      return new Response("", { status: 200 });
    }
    return slackResponse(buildHelpText());
  }

  // List all recipes
  if (command === "list") {
    return slackResponse(formatRecipeList());
  }

  // Status
  if (command === "status") {
    const [runs, usage] = await Promise.all([
      fetchWorkflowRuns(env),
      fetchActionsUsage(env),
    ]);
    if (runs.length === 0) {
      return slackResponse("No recent runs found.");
    }
    const blocks = buildStatusBlocks(runs, usage);
    return new Response(
      JSON.stringify({
        response_type: "ephemeral",
        text: "Recent rottengenizdat runs",
        blocks,
      }),
      { headers: { "Content-Type": "application/json" } },
    );
  }

  // No command -> open the modal
  if (!command && urls.length === 0) {
    const triggerId = params.get("trigger_id");
    const channelId = params.get("channel_id") ?? "";
    if (triggerId) {
      const error = await openModal(env, triggerId, channelId);
      if (error) {
        return slackResponse(`:x: Failed to open modal: ${error}`);
      }
    }
    return new Response("", { status: 200 });
  }

  // Explicit "random" or bare URLs dispatch directly
  if (!command || command === "random") {
    const [dispatched, usage] = await Promise.all([
      dispatchWorkflow(env, "", 3, "splice", urls),
      fetchActionsUsage(env),
    ]);
    if (dispatched) {
      const urlNote = urls.length > 0 ? ` with ${urls.length} URL(s)` : "";
      const msg = `:game_die: Firing up a random recipe${urlNote}...`;
      const blocks: object[] = [
        { type: "section", text: { type: "mrkdwn", text: msg } },
        {
          type: "actions",
          elements: [{
            type: "button",
            text: { type: "plain_text", text: "Check Status" },
            action_id: "modal_open_status",
          }],
        },
      ];
      if (usage) blocks.push(buildUsageContextShort(usage));
      return new Response(
        JSON.stringify({ response_type: "ephemeral", text: msg, blocks }),
        { headers: { "Content-Type": "application/json" } },
      );
    }
    return slackResponse(":warning: Failed to dispatch workflow. Check the GitHub token configuration.");
  }

  // Specific recipe
  if (isValidRecipe(command)) {
    const [dispatched, usage] = await Promise.all([
      dispatchWorkflow(env, command, 3, "splice", urls),
      fetchActionsUsage(env),
    ]);
    if (dispatched) {
      const urlNote = urls.length > 0 ? ` with ${urls.length} URL(s)` : "";
      const msg = `:radio: Firing up *${command}*${urlNote}...`;
      const blocks: object[] = [
        { type: "section", text: { type: "mrkdwn", text: msg } },
        {
          type: "actions",
          elements: [{
            type: "button",
            text: { type: "plain_text", text: "Check Status" },
            action_id: "modal_open_status",
          }],
        },
      ];
      if (usage) blocks.push(buildUsageContextShort(usage));
      return new Response(
        JSON.stringify({ response_type: "ephemeral", text: msg, blocks }),
        { headers: { "Content-Type": "application/json" } },
      );
    }
    return slackResponse(":warning: Failed to dispatch workflow. Check the GitHub token configuration.");
  }

  // Invalid recipe -- suggest similar ones
  const suggestions = suggestRecipes(command);
  let errorText = `:x: Unknown recipe \`${command}\`.`;
  if (suggestions.length > 0) {
    const suggestionList = suggestions.map((r) => `\`${r.slug}\``).join(", ");
    errorText += `\n\nDid you mean: ${suggestionList}?`;
  }
  errorText += "\n\nUse `/rottengenizdat list` to see all available recipes.";
  return slackResponse(errorText);
}

// ---------------------------------------------------------------------------
// Modal openers
// ---------------------------------------------------------------------------

async function openViewModal(env: Env, triggerId: string, view: object): Promise<string | null> {
  const resp = await fetch("https://slack.com/api/views.open", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.SLACK_BOT_TOKEN}`,
    },
    body: JSON.stringify({ trigger_id: triggerId, view }),
  });
  const data = (await resp.json()) as { ok: boolean; error?: string };
  if (!data.ok) {
    console.error(`views.open failed: ${data.error}`);
    return data.error ?? "unknown error";
  }
  return null;
}

async function openModal(env: Env, triggerId: string, channelId: string): Promise<string | null> {
  const view = buildModalView(channelId);
  return openViewModal(env, triggerId, view);
}

// ---------------------------------------------------------------------------
// Worker entry point
// ---------------------------------------------------------------------------

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "POST") {
      const body = await request.text();

      const valid = await verifySlackSignature(
        request,
        body,
        env.SLACK_SIGNING_SECRET,
      );
      if (!valid) {
        return new Response("Invalid signature", { status: 401 });
      }

      if (url.pathname === "/slack/interactions") {
        return handleInteraction(body, env, ctx);
      }

      if (url.pathname === "/slack/commands") {
        return handleSlashCommand(body, env, ctx);
      }
    }

    return new Response("Not Found", { status: 404 });
  },
};
