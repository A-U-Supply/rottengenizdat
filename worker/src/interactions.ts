import type { Env } from "./types";
import { dispatchWorkflow, fetchWorkflowRuns, fetchActionsUsage } from "./github";
import { buildUsageContextShort } from "./blocks";
import { buildHelpView, buildStatusView } from "./modal";
import { getRecipe } from "./recipes";

// ---------------------------------------------------------------------------
// Slack interaction payload types
// ---------------------------------------------------------------------------

interface SlackInteractionPayload {
  type: string;
  user: { id: string };
  actions?: Array<{
    action_id: string;
    value: string;
    block_id: string;
  }>;
  view?: {
    id: string;
    callback_id: string;
    state: { values: Record<string, Record<string, any>> };
    private_metadata: string;
  };
  trigger_id?: string;
}

// ---------------------------------------------------------------------------
// Slack API helpers
// ---------------------------------------------------------------------------

/** Build Block Kit blocks for a dispatch confirmation with a status button. */
function confirmationBlocks(text: string): object[] {
  return [
    {
      type: "section",
      text: { type: "mrkdwn", text },
    },
    {
      type: "actions",
      elements: [
        {
          type: "button",
          text: { type: "plain_text", text: "Check Status" },
          action_id: "modal_open_status",
        },
      ],
    },
  ];
}

/** Push a new view onto the Slack modal stack. */
async function pushView(
  env: Env,
  triggerId: string,
  view: object,
): Promise<void> {
  const resp = await fetch("https://slack.com/api/views.push", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.SLACK_BOT_TOKEN}`,
    },
    body: JSON.stringify({ trigger_id: triggerId, view }),
  });
  const data = (await resp.json()) as { ok: boolean; error?: string };
  if (!data.ok) {
    console.error(`views.push failed: ${data.error}`);
  }
}

/** Open a new modal view. */
async function openView(
  env: Env,
  triggerId: string,
  view: object,
): Promise<void> {
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
  }
}

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

/**
 * Handle Slack interactive payloads (button clicks, modal submissions).
 */
export async function handleInteraction(
  body: string,
  env: Env,
  ctx?: ExecutionContext,
): Promise<Response> {
  const params = new URLSearchParams(body);
  const payloadStr = params.get("payload");
  if (!payloadStr) {
    return new Response("Missing payload", { status: 400 });
  }

  let payload: SlackInteractionPayload;
  try {
    payload = JSON.parse(payloadStr);
  } catch {
    return new Response("Invalid JSON payload", { status: 400 });
  }

  switch (payload.type) {
    case "block_actions": {
      const actions = payload.actions || [];
      for (const action of actions) {
        switch (action.action_id) {
          case "modal_open_help": {
            if (payload.trigger_id) {
              const view = buildHelpView();
              await pushView(env, payload.trigger_id, view);
            }
            break;
          }
          case "modal_open_status": {
            if (payload.trigger_id) {
              const [runs, usage] = await Promise.all([
                fetchWorkflowRuns(env),
                fetchActionsUsage(env),
              ]);
              const view = buildStatusView(runs, usage);
              if (payload.view) {
                await pushView(env, payload.trigger_id, view);
              } else {
                await openView(env, payload.trigger_id, view);
              }
            }
            break;
          }
          default:
            break;
        }
      }
      return new Response("", { status: 200 });
    }

    case "view_submission": {
      if (payload.view?.callback_id === "rottengenizdat_run") {
        const vals = payload.view.state.values;
        const recipe =
          vals.recipe_block?.recipe_select?.selected_option?.value ?? null;
        const sampleCount = parseInt(
          vals.sample_count_block?.sample_count_select?.selected_option?.value ?? "3",
          10,
        );
        const inputMode =
          vals.input_mode_block?.input_mode_select?.selected_option?.value ?? "splice";
        const rawUrls: string = vals.urls_block?.audio_urls?.value ?? "";
        const urls = rawUrls
          .split(/[\s,]+/)
          .map((u: string) => u.trim())
          .filter((u: string) => /^https?:\/\//i.test(u));

        const resolvedRecipe =
          recipe && recipe !== "random" ? recipe : "";
        const recipeName =
          getRecipe(resolvedRecipe)?.name ?? (resolvedRecipe || "random");

        const channelId = payload.view.private_metadata;
        const userId = payload.user.id;

        // Dispatch in background — modal must respond immediately
        const work = (async () => {
          let ok = false;
          let dispatchError = "";
          let usage: Awaited<ReturnType<typeof fetchActionsUsage>> = null;

          try {
            const results = await Promise.allSettled([
              dispatchWorkflow(env, resolvedRecipe, sampleCount, inputMode, urls),
              fetchActionsUsage(env),
            ]);
            ok = results[0].status === "fulfilled" && results[0].value === true;
            if (results[0].status === "rejected") {
              dispatchError = String(results[0].reason);
            }
            if (results[1].status === "fulfilled") {
              usage = results[1].value;
            }
          } catch (err) {
            dispatchError = String(err);
          }

          // Always post feedback
          const postTo = channelId || undefined;
          if (postTo) {
            const msg = ok
              ? `:radio: Firing up *${recipeName}* with ${sampleCount} sample(s) (${inputMode})... results incoming.`
              : `:warning: Failed to dispatch workflow.${dispatchError ? ` Error: ${dispatchError}` : ""}`;
            const blocks = confirmationBlocks(msg);
            if (usage) blocks.push(buildUsageContextShort(usage));
            await fetch("https://slack.com/api/chat.postEphemeral", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${env.SLACK_BOT_TOKEN}`,
              },
              body: JSON.stringify({ channel: postTo, user: userId, text: msg, blocks }),
            });
          }
        })();
        if (ctx) ctx.waitUntil(work);
        else work.catch(console.error);
      }
      return new Response(JSON.stringify({ response_action: "clear" }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    case "block_suggestion": {
      return new Response(JSON.stringify({ options: [] }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    default:
      return new Response("", { status: 200 });
  }
}
