import { type AssistantMessage, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import type { AgentSessionEvent } from "../../../src/core/agent-session.ts";
import { createHarness, type Harness } from "../harness.ts";

type SessionWithRecoveryInternals = {
	_checkCompaction(message: AssistantMessage, skipAbortedCheck?: boolean): Promise<boolean>;
	_overflowRecoveryAttempted: boolean;
};

// Overflow/length recovery must not spend its single retry or write context_edit entries
// when a session_before_compact extension handler cancels the compaction.
describe("overflow recovery commits only after session_before_compact accepts", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	async function setup(cancel: boolean) {
		let hookCalls = 0;
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow: 500_000, maxTokens: 32_000 }],
			settings: { compaction: { enabled: true, reserveTokens: 16_000, keepRecentTokens: 20_000 } },
			extensionFactories: [
				(pi) => {
					pi.on("session_before_compact", async () => {
						hookCalls++;
						if (cancel) return { cancel: true };
						return undefined;
					});
				},
			],
		});
		harnesses.push(harness);

		const model = harness.getModel();
		const now = Date.now();
		const big = "x".repeat(40_000);
		for (let i = 0; i < 4; i++) {
			harness.sessionManager.appendMessage({ role: "user", content: big, timestamp: now + i * 10 });
			harness.sessionManager.appendMessage({
				...fauxAssistantMessage(big, { timestamp: now + i * 10 + 1 }),
				api: model.api,
				provider: model.provider,
				model: model.id,
			});
		}
		const failed: AssistantMessage = {
			...fauxAssistantMessage("cut", { stopReason: "length", timestamp: now + 100 }),
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 100,
				output: 1,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 101,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
		};
		harness.sessionManager.appendMessage(failed);
		harness.session.agent.state.messages = harness.sessionManager.buildSessionContext().messages;

		const events: AgentSessionEvent[] = [];
		harness.session.subscribe((event) => events.push(event));
		const internals = harness.session as unknown as SessionWithRecoveryInternals;
		const contextEdits = () => harness.sessionManager.getEntries().filter((entry) => entry.type === "context_edit");
		return { harness, failed, internals, events, contextEdits, hookCalls: () => hookCalls };
	}

	it("a cancelled hook leaves the retry budget and session entries untouched", async () => {
		const { failed, internals, events, contextEdits } = await setup(true);

		const retry = await internals._checkCompaction(failed);

		expect(retry).toBe(false);
		expect(contextEdits()).toHaveLength(0);
		expect(internals._overflowRecoveryAttempted).toBe(false);
		const ends = events.filter((event) => event.type === "compaction_end");
		expect(ends).toHaveLength(1);
		expect(ends[0]).toMatchObject({ aborted: true, errorMessage: undefined });
	});

	it("a later length stop after a cancelled recovery is handled again", async () => {
		const { failed, internals, events, hookCalls } = await setup(true);

		await internals._checkCompaction(failed);
		await internals._checkCompaction(failed);

		expect(hookCalls()).toBe(2);
		const ends = events.filter((event) => event.type === "compaction_end");
		expect(ends).toHaveLength(2);
		expect(ends.every((event) => event.type === "compaction_end" && event.errorMessage === undefined)).toBe(true);
	});

	it("an accepted hook commits the recovery and summarizes the omitted branch", async () => {
		const { harness, failed, internals, contextEdits } = await setup(false);
		let summaryRequest = "";
		harness.setResponses([
			(context) => {
				summaryRequest = JSON.stringify(context.messages);
				return fauxAssistantMessage("summary");
			},
		]);

		const retry = await internals._checkCompaction(failed);

		expect(retry).toBe(true);
		expect(contextEdits()).toHaveLength(1);
		expect(internals._overflowRecoveryAttempted).toBe(true);
		expect(summaryRequest).not.toBe("");
		expect(summaryRequest).not.toContain("cut");
	});
});
