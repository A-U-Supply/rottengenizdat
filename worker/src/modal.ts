import { RECIPES, getRecipe } from "./recipes";
import type { Recipe } from "./recipes";
import type { ActionsUsage, WorkflowRun } from "./types";
import { buildStatusBlocks } from "./blocks";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Truncate a string, adding ellipsis if needed. */
function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 1) + "\u2026";
}

/** Build static option_groups for the recipe selector, grouped by mode. */
function buildRecipeOptionGroups(): object[] {
  const groups: Array<{ label: { type: string; text: string }; options: object[] }> = [];

  // Random option first
  groups.push({
    label: { type: "plain_text", text: "Special" },
    options: [{ text: { type: "plain_text", text: "\ud83c\udfb2 Random" }, value: "random" }],
  });

  // Group by mode
  const sequential = RECIPES.filter((r) => r.mode === "sequential");
  const branch = RECIPES.filter((r) => r.mode === "branch");

  if (sequential.length > 0) {
    groups.push({
      label: { type: "plain_text", text: "Sequential (stacked)" },
      options: sequential.map((r) => ({
        text: { type: "plain_text", text: truncate(`${r.name} (${r.steps} steps)`, 75) },
        value: r.slug,
      })),
    });
  }

  if (branch.length > 0) {
    groups.push({
      label: { type: "plain_text", text: "Branch (parallel mix)" },
      options: branch.map((r) => ({
        text: { type: "plain_text", text: truncate(`${r.name} (${r.steps} steps)`, 75) },
        value: r.slug,
      })),
    });
  }

  return groups;
}

// ---------------------------------------------------------------------------
// Info modal views
// ---------------------------------------------------------------------------

/** Build the comprehensive Help modal. */
export function buildHelpView(): object {
  const blocks: object[] = [];

  blocks.push({
    type: "header",
    text: { type: "plain_text", text: "Quick Start", emoji: true },
  });
  blocks.push({
    type: "section",
    text: {
      type: "mrkdwn",
      text: [
        "Type `/rottengenizdat` and hit enter to open the recipe picker.",
        "Choose a recipe (or leave it on Random), set how many samples to pull, and hit *Mangle*.",
        "Results appear in the channel in ~5\u201310 minutes.",
      ].join("\n"),
    },
  });

  blocks.push({ type: "divider" });

  blocks.push({
    type: "header",
    text: { type: "plain_text", text: "Commands", emoji: true },
  });
  blocks.push({
    type: "section",
    text: {
      type: "mrkdwn",
      text: [
        "`/rottengenizdat` \u2014 Open the recipe picker modal",
        "`/rottengenizdat custom` \u2014 Open the custom chain builder",
        "`/rottengenizdat <recipe>` \u2014 Run a specific recipe directly",
        "`/rottengenizdat list` \u2014 Show all recipes grouped by mode",
        "`/rottengenizdat status` \u2014 Recent workflow run status",
        "`/rottengenizdat help` \u2014 This help screen",
      ].join("\n"),
    },
  });

  blocks.push({ type: "divider" });

  blocks.push({
    type: "header",
    text: { type: "plain_text", text: "The Modal", emoji: true },
  });
  blocks.push({
    type: "section",
    text: {
      type: "mrkdwn",
      text: [
        "*Recipe* \u2014 Pick a recipe or leave on Random. Type to search.",
        "*Sample count* \u2014 How many random audio clips to pull from #sample-sale.",
        "*Input mode* \u2014 How multiple inputs are combined: splice (chop & shuffle), concat (end-to-end), blend (each through recipe, mix outputs), or independent (each processed separately).",
        "*Audio URLs* \u2014 Paste audio/video URLs directly instead of (or in addition to) #sample-sale samples.",
      ].join("\n"),
    },
  });

  blocks.push({ type: "divider" });

  blocks.push({
    type: "header",
    text: { type: "plain_text", text: "How It Works", emoji: true },
  });
  blocks.push({
    type: "section",
    text: {
      type: "mrkdwn",
      text: [
        "1. Your command triggers a GitHub Actions pipeline",
        "2. Random audio samples are pulled from #sample-sale (or your URLs)",
        "3. Inputs are combined (splice/concat/blend/independent) and fed through RAVE neural networks",
        "4. The recipe's effect chain mangles the audio through one or more models",
        "5. The result is posted back to the channel",
      ].join("\n"),
    },
  });

  blocks.push({ type: "divider" });

  blocks.push({
    type: "header",
    text: { type: "plain_text", text: "Recipes", emoji: true },
  });
  blocks.push({
    type: "section",
    text: {
      type: "mrkdwn",
      text: [
        `${RECIPES.length} recipes available, ranging from barely-noticeable to total sonic destruction.`,
        "",
        "*Sequential* recipes feed audio through each step in order \u2014 a photocopy of a photocopy.",
        "*Branch* recipes send the original to every step independently, then mix all outputs together.",
        "",
        "Use `/rottengenizdat list` to see them all.",
      ].join("\n"),
    },
  });

  return {
    type: "modal",
    callback_id: "rottengenizdat_help",
    title: { type: "plain_text", text: "Help" },
    close: { type: "plain_text", text: "Back" },
    blocks,
  };
}

/** Build the Status modal showing recent workflow runs. */
export function buildStatusView(runs: WorkflowRun[], usage?: ActionsUsage | null): object {
  const blocks: object[] = runs.length > 0
    ? buildStatusBlocks(runs, usage)
    : [{
        type: "section",
        text: { type: "mrkdwn", text: "No recent runs found." },
      }];

  return {
    type: "modal",
    callback_id: "rottengenizdat_status",
    title: { type: "plain_text", text: "Status" },
    close: { type: "plain_text", text: "Back" },
    blocks,
  };
}

// ---------------------------------------------------------------------------
// Custom modal helpers
// ---------------------------------------------------------------------------

const MODELS = [
  "percussion", "vintage", "VCTK", "nasa", "musicnet",
  "isis", "sol_ordinario", "sol_full", "sol_ordinario_fast", "darbouka_onnx",
];

const TEMPERATURE_OPTIONS = ["0.3", "0.5", "0.7", "0.9", "1.0", "1.2", "1.5", "2.0", "2.5"];
const NOISE_OPTIONS = ["0.0", "0.1", "0.2", "0.3", "0.5", "0.8"];
const MIX_OPTIONS = ["0.3", "0.5", "0.7", "1.0"];
const SHUFFLE_OPTIONS = ["2", "3", "4", "6", "8"];
const QUANTIZE_OPTIONS = ["0.1", "0.3", "0.5", "0.7", "1.0"];

function opt(text: string, value?: string): object {
  return { text: { type: "plain_text", text }, value: value ?? text };
}

/** Build blocks for a single effect step in the custom modal. */
function buildStepBlocks(stepNum: number, required: boolean): object[] {
  const prefix = `step${stepNum}`;
  const blocks: object[] = [];

  // Step header
  blocks.push({
    type: "header",
    text: { type: "plain_text", text: `Step ${stepNum}${required ? "" : " (optional)"}` },
  });

  // Model select
  const modelOptions = required
    ? MODELS.map((m) => opt(m))
    : [opt("— skip —", "none"), ...MODELS.map((m) => opt(m))];
  blocks.push({
    type: "input",
    block_id: `${prefix}_model_block`,
    optional: !required,
    label: { type: "plain_text", text: "Model" },
    element: {
      type: "static_select",
      action_id: `${prefix}_model`,
      ...(required
        ? { initial_option: opt("vintage") }
        : { initial_option: opt("— skip —", "none") }),
      options: modelOptions,
    },
  });

  // Temperature
  blocks.push({
    type: "input",
    block_id: `${prefix}_temp_block`,
    optional: true,
    label: { type: "plain_text", text: "Temperature" },
    element: {
      type: "static_select",
      action_id: `${prefix}_temp`,
      initial_option: opt("1.0"),
      options: TEMPERATURE_OPTIONS.map((v) => opt(v)),
    },
  });

  // Noise
  blocks.push({
    type: "input",
    block_id: `${prefix}_noise_block`,
    optional: true,
    label: { type: "plain_text", text: "Noise" },
    element: {
      type: "static_select",
      action_id: `${prefix}_noise`,
      initial_option: opt("0.0"),
      options: NOISE_OPTIONS.map((v) => opt(v)),
    },
  });

  // Mix (wet/dry)
  blocks.push({
    type: "input",
    block_id: `${prefix}_mix_block`,
    optional: true,
    label: { type: "plain_text", text: "Mix (wet/dry)" },
    element: {
      type: "static_select",
      action_id: `${prefix}_mix`,
      initial_option: opt("1.0"),
      options: MIX_OPTIONS.map((v) => opt(v)),
    },
  });

  // Dims (multi-select)
  blocks.push({
    type: "input",
    block_id: `${prefix}_dims_block`,
    optional: true,
    label: { type: "plain_text", text: "Dims (all if none selected)" },
    element: {
      type: "multi_static_select",
      action_id: `${prefix}_dims`,
      placeholder: { type: "plain_text", text: "Select dims..." },
      options: Array.from({ length: 16 }, (_, i) => opt(String(i))),
    },
  });

  // Reverse + Shuffle + Quantize on one row would be nice but Slack doesn't
  // support inline inputs, so they each get their own block.

  // Reverse (checkbox)
  blocks.push({
    type: "input",
    block_id: `${prefix}_reverse_block`,
    optional: true,
    label: { type: "plain_text", text: "Reverse latent" },
    element: {
      type: "checkboxes",
      action_id: `${prefix}_reverse`,
      options: [opt("Reverse", "true")],
    },
  });

  // Shuffle chunks
  blocks.push({
    type: "input",
    block_id: `${prefix}_shuffle_block`,
    optional: true,
    label: { type: "plain_text", text: "Shuffle chunks" },
    element: {
      type: "static_select",
      action_id: `${prefix}_shuffle`,
      placeholder: { type: "plain_text", text: "None" },
      options: SHUFFLE_OPTIONS.map((v) => opt(v)),
    },
  });

  // Quantize
  blocks.push({
    type: "input",
    block_id: `${prefix}_quantize_block`,
    optional: true,
    label: { type: "plain_text", text: "Quantize" },
    element: {
      type: "static_select",
      action_id: `${prefix}_quantize`,
      placeholder: { type: "plain_text", text: "None" },
      options: QUANTIZE_OPTIONS.map((v) => opt(v)),
    },
  });

  return blocks;
}

/** Build the Custom modal view with up to 3 effect steps and all knobs. */
export function buildCustomModalView(channelId: string = ""): object {
  const blocks: object[] = [];

  // Description
  blocks.push({
    type: "section",
    text: {
      type: "mrkdwn",
      text: "Build a custom effect chain. Pick models, tweak every knob, chain up to 3 steps.",
    },
  });
  blocks.push({ type: "divider" });

  // Chain mode
  blocks.push({
    type: "input",
    block_id: "chain_mode_block",
    label: { type: "plain_text", text: "Chain mode" },
    element: {
      type: "static_select",
      action_id: "chain_mode_select",
      initial_option: opt("Sequential (each step feeds the next)", "sequential"),
      options: [
        opt("Sequential (each step feeds the next)", "sequential"),
        opt("Branch (all steps in parallel, mix outputs)", "branch"),
      ],
    },
  });

  // Steps
  blocks.push(...buildStepBlocks(1, true));
  blocks.push({ type: "divider" });
  blocks.push(...buildStepBlocks(2, false));
  blocks.push({ type: "divider" });
  blocks.push(...buildStepBlocks(3, false));
  blocks.push({ type: "divider" });

  // Input options header
  blocks.push({
    type: "header",
    text: { type: "plain_text", text: "Input" },
  });

  // Sample count
  blocks.push({
    type: "input",
    block_id: "custom_sample_count_block",
    label: { type: "plain_text", text: "Sample count" },
    element: {
      type: "static_select",
      action_id: "custom_sample_count_select",
      initial_option: opt("3"),
      options: [opt("1"), opt("2"), opt("3"), opt("4"), opt("5")],
    },
  });

  // Input mode
  blocks.push({
    type: "input",
    block_id: "custom_input_mode_block",
    label: { type: "plain_text", text: "Input mode" },
    element: {
      type: "static_select",
      action_id: "custom_input_mode_select",
      initial_option: opt("Splice (chop & shuffle)", "splice"),
      options: [
        opt("Splice (chop & shuffle)", "splice"),
        opt("Concat (end-to-end)", "concat"),
        opt("Blend (each through chain, mix outputs)", "blend"),
        opt("Independent (each separately)", "independent"),
      ],
    },
  });

  // Audio URLs
  blocks.push({
    type: "input",
    block_id: "custom_urls_block",
    optional: true,
    label: { type: "plain_text", text: "Audio URLs" },
    element: {
      type: "plain_text_input",
      action_id: "custom_audio_urls",
      multiline: true,
      placeholder: { type: "plain_text", text: "Paste URLs, one per line (optional)" },
    },
  });

  // Footer
  blocks.push({ type: "divider" });
  blocks.push({
    type: "actions",
    block_id: "custom_footer_actions",
    elements: [
      {
        type: "button",
        text: { type: "plain_text", text: "\ud83d\udce1 Status", emoji: true },
        action_id: "modal_open_status",
      },
      {
        type: "button",
        text: { type: "plain_text", text: "\u2753 Help", emoji: true },
        action_id: "modal_open_help",
      },
    ],
  });

  return {
    type: "modal",
    callback_id: "rottengenizdat_custom",
    private_metadata: channelId,
    title: { type: "plain_text", text: "Custom Chain" },
    submit: { type: "plain_text", text: "Mangle" },
    close: { type: "plain_text", text: "Cancel" },
    blocks,
  };
}

// ---------------------------------------------------------------------------
// Main modal view builder
// ---------------------------------------------------------------------------

/** Build the Slack modal view object for `views.open`. */
export function buildModalView(channelId: string = ""): object {
  const RANDOM_OPTION = { text: { type: "plain_text", text: "\ud83c\udfb2 Random" }, value: "random" };

  return {
    type: "modal",
    callback_id: "rottengenizdat_run",
    private_metadata: channelId,
    title: { type: "plain_text", text: "rottengenizdat" },
    submit: { type: "plain_text", text: "Mangle" },
    close: { type: "plain_text", text: "Cancel" },
    blocks: [
      // Description
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: "Pick a recipe and hit *Mangle* to feed audio from #sample-sale through RAVE neural networks. Bone music for the machine age.",
        },
      },
      { type: "divider" },
      // Recipe select
      {
        type: "input",
        block_id: "recipe_block",
        optional: true,
        label: { type: "plain_text", text: "Recipe" },
        element: {
          type: "static_select",
          action_id: "recipe_select",
          initial_option: RANDOM_OPTION,
          placeholder: { type: "plain_text", text: "Search recipes\u2026" },
          option_groups: buildRecipeOptionGroups(),
        },
      },
      // Sample count
      {
        type: "input",
        block_id: "sample_count_block",
        label: { type: "plain_text", text: "Sample count" },
        element: {
          type: "static_select",
          action_id: "sample_count_select",
          initial_option: { text: { type: "plain_text", text: "3" }, value: "3" },
          options: [
            { text: { type: "plain_text", text: "1" }, value: "1" },
            { text: { type: "plain_text", text: "2" }, value: "2" },
            { text: { type: "plain_text", text: "3" }, value: "3" },
            { text: { type: "plain_text", text: "4" }, value: "4" },
            { text: { type: "plain_text", text: "5" }, value: "5" },
          ],
        },
      },
      // Input mode
      {
        type: "input",
        block_id: "input_mode_block",
        label: { type: "plain_text", text: "Input mode" },
        element: {
          type: "static_select",
          action_id: "input_mode_select",
          initial_option: {
            text: { type: "plain_text", text: "Splice (chop & shuffle)" },
            value: "splice",
          },
          options: [
            {
              text: { type: "plain_text", text: "Splice (chop & shuffle)" },
              value: "splice",
            },
            {
              text: { type: "plain_text", text: "Concat (end-to-end)" },
              value: "concat",
            },
            {
              text: { type: "plain_text", text: "Blend (each through recipe, mix outputs)" },
              value: "blend",
            },
            {
              text: { type: "plain_text", text: "Independent (each separately)" },
              value: "independent",
            },
          ],
        },
      },
      // Audio URLs (optional)
      {
        type: "input",
        block_id: "urls_block",
        optional: true,
        label: { type: "plain_text", text: "Audio URLs" },
        hint: {
          type: "plain_text",
          text: "Direct links to audio/video files. Used in addition to (or instead of) #sample-sale samples.",
        },
        element: {
          type: "plain_text_input",
          action_id: "audio_urls",
          multiline: true,
          placeholder: {
            type: "plain_text",
            text: "Paste URLs, one per line (optional)",
          },
        },
      },
      // Footer
      { type: "divider" },
      {
        type: "context",
        elements: [{ type: "mrkdwn", text: `_${RECIPES.length} recipes available_` }],
      },
      {
        type: "actions",
        block_id: "modal_footer_actions",
        elements: [
          {
            type: "button",
            text: { type: "plain_text", text: "\ud83d\udd27 Custom", emoji: true },
            action_id: "modal_open_custom",
          },
          {
            type: "button",
            text: { type: "plain_text", text: "\ud83d\udce1 Status", emoji: true },
            action_id: "modal_open_status",
          },
          {
            type: "button",
            text: { type: "plain_text", text: "\u2753 Help", emoji: true },
            action_id: "modal_open_help",
          },
        ],
      },
    ],
  };
}
