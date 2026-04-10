import type { ParsedCommand } from "./types";

// ---------------------------------------------------------------------------
// Slash command parsing
// ---------------------------------------------------------------------------

/**
 * Parse a Slack slash command text field into a command and optional URLs.
 *
 * The first non-URL token is the command (lowercased). All tokens matching
 * http(s):// are collected as URLs with original case preserved.
 */
export function parseSlashCommand(text: string): ParsedCommand {
  const parts = text.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) {
    return { command: "", urls: [] };
  }

  const first = parts[0];
  const isUrl = /^https?:\/\//i;

  if (isUrl.test(first)) {
    return {
      command: "",
      urls: parts.filter((s) => isUrl.test(s)),
    };
  }

  return {
    command: first.toLowerCase(),
    urls: parts.slice(1).filter((s) => isUrl.test(s)),
  };
}

// ---------------------------------------------------------------------------
// Slack signature verification
// ---------------------------------------------------------------------------

/**
 * Verify the Slack request signature using HMAC-SHA256.
 */
export async function verifySlackSignature(
  request: Request,
  body: string,
  signingSecret: string,
): Promise<boolean> {
  const timestamp = request.headers.get("x-slack-request-timestamp");
  const slackSignature = request.headers.get("x-slack-signature");

  if (!timestamp || !slackSignature) {
    return false;
  }

  const now = Math.floor(Date.now() / 1000);
  const ts = Number(timestamp);
  if (Number.isNaN(ts) || Math.abs(now - ts) > 300) {
    return false;
  }

  const sigBasestring = `v0:${timestamp}:${body}`;

  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(signingSecret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );

  const signatureBytes = await crypto.subtle.sign(
    "HMAC",
    key,
    encoder.encode(sigBasestring),
  );

  const expectedSignature =
    "v0=" +
    [...new Uint8Array(signatureBytes)]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");

  if (expectedSignature.length !== slackSignature.length) {
    return false;
  }
  let mismatch = 0;
  for (let i = 0; i < expectedSignature.length; i++) {
    mismatch |= expectedSignature.charCodeAt(i) ^ slackSignature.charCodeAt(i);
  }
  return mismatch === 0;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a JSON response that Slack understands. */
export function slackResponse(text: string, ephemeral = true): Response {
  return new Response(
    JSON.stringify({
      response_type: ephemeral ? "ephemeral" : "in_channel",
      text,
    }),
    {
      headers: { "Content-Type": "application/json" },
    },
  );
}
