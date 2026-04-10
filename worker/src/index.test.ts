import { describe, it, expect } from "vitest";
import { parseSlashCommand } from "./slack";

describe("parseSlashCommand", () => {
  it("returns empty command for empty text", () => {
    const result = parseSlashCommand("");
    expect(result.command).toBe("");
    expect(result.urls).toEqual([]);
  });

  it("parses a recipe name", () => {
    const result = parseSlashCommand("bone-xray");
    expect(result.command).toBe("bone-xray");
    expect(result.urls).toEqual([]);
  });

  it("parses a recipe name with URLs", () => {
    const result = parseSlashCommand("fever-dream https://example.com/audio.wav");
    expect(result.command).toBe("fever-dream");
    expect(result.urls).toEqual(["https://example.com/audio.wav"]);
  });

  it("treats leading URL as no command", () => {
    const result = parseSlashCommand("https://example.com/a.wav https://example.com/b.wav");
    expect(result.command).toBe("");
    expect(result.urls).toEqual(["https://example.com/a.wav", "https://example.com/b.wav"]);
  });

  it("lowercases command", () => {
    const result = parseSlashCommand("HELP");
    expect(result.command).toBe("help");
  });

  it("handles whitespace-only input", () => {
    const result = parseSlashCommand("   ");
    expect(result.command).toBe("");
    expect(result.urls).toEqual([]);
  });

  it("parses multiple URLs after command", () => {
    const result = parseSlashCommand("bone-xray https://a.com/1.wav https://b.com/2.wav");
    expect(result.command).toBe("bone-xray");
    expect(result.urls).toEqual(["https://a.com/1.wav", "https://b.com/2.wav"]);
  });
});
