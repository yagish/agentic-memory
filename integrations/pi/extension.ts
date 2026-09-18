import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Box, Text } from "@earendil-works/pi-tui";
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const EXT_DIR = dirname(fileURLToPath(import.meta.url));
const ADAPTER_PATH = join(EXT_DIR, "adapter.py");
const PYTHON_BIN = process.env.MEMORY_PYTHON ?? "python3";
const MESSAGE_TYPE = "agentic-memory";
const EXTENSION_SINGLETON_KEY = "__agenticMemoryExtensionRegistered";

type ExtensionGlobalState = typeof globalThis & {
	[EXTENSION_SINGLETON_KEY]?: boolean;
};

type SavedTurn = { role: "user" | "assistant"; content: string };
type RecallResponse =
	| { action: "noop"; warnings?: Array<{ stage: string; message: string }> }
	| { action: "answer"; answer: string; warnings?: Array<{ stage: string; message: string }> }
	| { action: "inject"; injection: string; warnings?: Array<{ stage: string; message: string }> };

function extractTextContent(content: unknown): string {
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";
	return content
		.filter((block): block is { type: string; text?: string } => typeof block === "object" && block !== null && "type" in block)
		.filter((block) => block.type === "text" && typeof block.text === "string")
		.map((block) => block.text!.trim())
		.filter(Boolean)
		.join("\n")
		.trim();
}

function buildTurnsFromBranch(ctx: ExtensionContext): { turns: SavedTurn[]; startedAt?: string } {
	const turns: SavedTurn[] = [];
	let startedAt: string | undefined;

	for (const entry of ctx.sessionManager.getBranch() as Array<any>) {
		if (entry.type !== "message") continue;
		const message = entry.message;
		if (!message || typeof message !== "object") continue;

		if (message.role === "user") {
			const content = extractTextContent(message.content);
			if (!content) continue;
			turns.push({ role: "user", content });
			startedAt ??= new Date(message.timestamp ?? Date.now()).toISOString();
			continue;
		}

		if (message.role === "assistant") {
			if (message.stopReason === "toolUse") continue;
			const content = extractTextContent(message.content);
			if (!content) continue;
			turns.push({ role: "assistant", content });
			startedAt ??= new Date(message.timestamp ?? Date.now()).toISOString();
		}
	}

	return { turns, startedAt };
}

function buildSavePayload(ctx: ExtensionContext, extraTurns: SavedTurn[] = []) {
	const { turns, startedAt } = buildTurnsFromBranch(ctx);
	const allTurns = [...turns, ...extraTurns];
	if (allTurns.length === 0) return null;

	return {
		session_id: ctx.sessionManager.getSessionId(),
		agent: "pi",
		turns: allTurns,
		started_at: startedAt,
		updated_at: new Date().toISOString(),
		metadata: {
			integration: "pi",
			cwd: ctx.cwd,
			session_file: ctx.sessionManager.getSessionFile(),
		},
	};
}

function callAdapter<T>(command: "save" | "recall", payload: unknown, signal?: AbortSignal): Promise<T> {
	return new Promise((resolve, reject) => {
		const child = spawn(PYTHON_BIN, [ADAPTER_PATH, command], {
			stdio: ["pipe", "pipe", "pipe"],
		});

		let stdout = "";
		let stderr = "";
		child.stdout.on("data", (chunk) => {
			stdout += String(chunk);
		});
		child.stderr.on("data", (chunk) => {
			stderr += String(chunk);
		});
		child.on("error", reject);
		child.on("close", (code) => {
			if (code !== 0) {
				reject(new Error(stderr.trim() || `adapter exited with code ${code}`));
				return;
			}
			try {
				resolve(JSON.parse(stdout || "{}") as T);
			} catch (error) {
				reject(new Error(`invalid adapter JSON: ${String(error)} :: ${stdout}`));
			}
		});

		if (signal) {
			const abortHandler = () => child.kill("SIGTERM");
			if (signal.aborted) abortHandler();
			signal.addEventListener("abort", abortHandler, { once: true });
		}

		child.stdin.write(JSON.stringify(payload));
		child.stdin.end();
	});
}

export default function agenticMemoryExtension(pi: ExtensionAPI) {
	const globalState = globalThis as ExtensionGlobalState;
	if (globalState[EXTENSION_SINGLETON_KEY]) {
		console.warn("agentic-memory extension already registered; skipping duplicate load");
		return;
	}
	globalState[EXTENSION_SINGLETON_KEY] = true;

	let pendingInjection: { prompt: string; injection: string } | null = null;

	pi.registerCommand("memory-status", {
		description: "Show agentic-memory extension status",
		handler: async (_args, ctx) => {
			try {
				const result = await callAdapter<{ action?: string; ok?: boolean; error?: string }>("recall", { prompt: "" });
				ctx.ui.notify(`agentic-memory loaded (${result.action ?? (result.ok ? "ok" : "unknown")})`, "info");
			} catch (error) {
				ctx.ui.notify(`agentic-memory loaded, adapter error: ${String(error)}`, "warning");
			}
		},
	});

	pi.registerMessageRenderer(MESSAGE_TYPE, (message, { outputPad }, theme) => {
		const details = (message.details ?? {}) as { kind?: "answer" | "context" };
		const label = details.kind === "answer" ? "Memory answer" : "Memory context";
		const box = new Box(outputPad, 1, (text) => theme.bg("customMessageBg", text));
		box.addChild(new Text(`${theme.fg("accent", `[${label}]`)} ${String(message.content)}`, 0, 0));
		return box;
	});

	pi.on("input", async (event, ctx) => {
		if (event.source === "extension") return { action: "continue" };

		pendingInjection = null;
		try {
			const result = await callAdapter<RecallResponse>(
				"recall",
				{
					session_id: ctx.sessionManager.getSessionId(),
					prompt: event.text,
					include_working_memory: false,
				},
			);

			if (result.action === "answer") {
				pi.sendMessage({
					customType: MESSAGE_TYPE,
					content: result.answer,
					display: true,
					details: { kind: "answer" },
				});

				const payload = buildSavePayload(ctx, [
					{ role: "user", content: event.text },
					{ role: "assistant", content: result.answer },
				]);
				if (payload) {
					try {
						await callAdapter("save", payload);
					} catch (error) {
						console.error("agentic-memory save failed:", error);
					}
				}
				return { action: "handled" };
			}

			if (result.action === "inject") {
				pendingInjection = { prompt: event.text, injection: result.injection };
			}
		} catch (error) {
			console.error("agentic-memory recall failed:", error);
		}

		return { action: "continue" };
	});

	pi.on("before_agent_start", async (event) => {
		if (!pendingInjection || pendingInjection.prompt !== event.prompt) return;
		const injection = pendingInjection.injection;
		pendingInjection = null;
		return {
			message: {
				customType: MESSAGE_TYPE,
				content: injection,
				display: true,
				details: { kind: "context" },
			},
		};
	});

	pi.on("turn_end", async (_event, ctx) => {
		const payload = buildSavePayload(ctx);
		if (!payload) return;
		try {
			await callAdapter("save", payload);
		} catch (error) {
			console.error("agentic-memory save failed:", error);
		}
	});
}
