import type Anthropic from "@anthropic-ai/sdk";
import type { ResponseStreamEvent } from "openai/resources/responses/responses.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamAnthropic } from "../src/api/anthropic-messages.ts";
import { processResponsesStream } from "../src/api/openai-responses-shared.ts";
import { getModel, normalizeContext } from "../src/compat.ts";
import type { AssistantMessage, AssistantMessageEvent, Model, ToolCall } from "../src/types.ts";
import { AssistantMessageEventStream } from "../src/utils/event-stream.ts";

const parseCalls = vi.hoisted(() => ({ streaming: 0 }));

vi.mock("../src/utils/json-parse.ts", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../src/utils/json-parse.ts")>();
	return {
		...actual,
		parseStreamingJson: (...args: Parameters<typeof actual.parseStreamingJson>) => {
			parseCalls.streaming++;
			return actual.parseStreamingJson(...args);
		},
	};
});

// Parsing the whole accumulated buffer on every delta is quadratic in the size of one tool call.
const CHUNK_SIZE = 64;
const LINE = 'line of text with "quotes", \\ backslashes and unicode \u00e9\n';

function bigArguments(lines: number): { text: string; json: string; chunks: string[] } {
	const text = LINE.repeat(lines);
	const json = JSON.stringify({ path: "/tmp/big.txt", content: text });
	const chunks: string[] = [];
	for (let i = 0; i < json.length; i += CHUNK_SIZE) chunks.push(json.slice(i, i + CHUNK_SIZE));
	return { text, json, chunks };
}

function createAnthropicClient(chunks: string[]): Anthropic {
	const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
	let body = sse("message_start", {
		type: "message_start",
		message: { id: "msg_test", usage: { input_tokens: 1, output_tokens: 0 } },
	});
	body += sse("content_block_start", {
		type: "content_block_start",
		index: 0,
		content_block: { type: "tool_use", id: "toolu_1", name: "write", input: {} },
	});
	for (const chunk of chunks) {
		body += sse("content_block_delta", {
			type: "content_block_delta",
			index: 0,
			delta: { type: "input_json_delta", partial_json: chunk },
		});
	}
	body += sse("content_block_stop", { type: "content_block_stop", index: 0 });
	body += sse("message_delta", {
		type: "message_delta",
		delta: { stop_reason: "tool_use" },
		usage: { input_tokens: 1, output_tokens: 2 },
	});
	body += sse("message_stop", { type: "message_stop" });
	return {
		beta: {
			messages: {
				create: () => ({
					asResponse: async () =>
						new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
				}),
			},
		},
	} as unknown as Anthropic;
}

async function runAnthropic(chunks: string[]) {
	const model = getModel("anthropic", "claude-haiku-4-5");
	const stream = streamAnthropic(
		model,
		normalizeContext({ messages: [{ role: "user", content: "go", timestamp: 1 }] }),
		{ client: createAnthropicClient(chunks) },
	);
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) events.push(event);
	return { events, result: await stream.result() };
}

function createResponsesOutput(model: Model<"openai-responses">): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "pending",
		timestamp: Date.now(),
	};
}

async function* responsesEvents(chunks: string[], json: string): AsyncIterable<ResponseStreamEvent> {
	const item = { type: "function_call", id: "fc_test", call_id: "call_test", name: "write" };
	yield {
		type: "response.output_item.added",
		output_index: 0,
		item: { ...item, arguments: "" },
	} as ResponseStreamEvent;
	for (const delta of chunks) {
		yield { type: "response.function_call_arguments.delta", output_index: 0, delta } as ResponseStreamEvent;
	}
	yield { type: "response.function_call_arguments.done", output_index: 0, arguments: json } as ResponseStreamEvent;
	yield {
		type: "response.output_item.done",
		output_index: 0,
		item: { ...item, arguments: json },
	} as ResponseStreamEvent;
	yield {
		type: "response.completed",
		sequence_number: 1,
		response: { id: "resp_test", status: "completed" },
	} as ResponseStreamEvent;
}

async function runResponses(chunks: string[], json: string) {
	const model: Model<"openai-responses"> = {
		id: "gpt-5-mini",
		name: "GPT-5 Mini",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://api.openai.com/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 400000,
		maxTokens: 128000,
	};
	const output = createResponsesOutput(model);
	const stream = new AssistantMessageEventStream();
	const pushSpy = vi.spyOn(stream, "push");
	await processResponsesStream(responsesEvents(chunks, json), output, stream, model);
	const events = pushSpy.mock.calls.map(([event]) => event);
	return { output, events };
}

describe("bounded streaming tool-call argument parsing", () => {
	beforeEach(() => {
		parseCalls.streaming = 0;
	});

	describe("anthropic-messages", () => {
		it("parses a large tool call a bounded number of times and ends with exact arguments", async () => {
			const { text, json, chunks } = bigArguments(2500);
			expect(json.length).toBeGreaterThan(100_000);

			const { events, result } = await runAnthropic(chunks);

			const call = result.content.find((block): block is ToolCall => block.type === "toolCall");
			expect(call?.arguments).toEqual({ path: "/tmp/big.txt", content: text });
			expect(call).not.toHaveProperty("partialJson");
			expect(call).not.toHaveProperty("lastParsedSize");
			expect(parseCalls.streaming).toBeLessThan(1000);
			// Every delta is still forwarded; only the parse is bounded.
			const deltas = events.filter((event) => event.type === "toolcall_delta").map((event) => event.delta);
			expect(deltas).toEqual(chunks);
			expect(events.at(-1)?.type).toBe("done");
		});

		it("still parses every delta of a small tool call", async () => {
			const { json, chunks } = bigArguments(1);
			const { result } = await runAnthropic(chunks);

			const call = result.content.find((block): block is ToolCall => block.type === "toolCall");
			expect(call?.arguments).toEqual(JSON.parse(json));
			expect(parseCalls.streaming).toBe(chunks.length);
		});
	});

	describe("openai-responses", () => {
		it("parses a large tool call a bounded number of times and ends with exact arguments", async () => {
			const { text, json, chunks } = bigArguments(2500);
			expect(json.length).toBeGreaterThan(100_000);

			const { output, events } = await runResponses(chunks, json);

			const call = output.content[0] as ToolCall;
			expect(call.arguments).toEqual({ path: "/tmp/big.txt", content: text });
			expect(call).not.toHaveProperty("partialJson");
			expect(call).not.toHaveProperty("lastParsedSize");
			// arguments.done and output_item.done parse once each, on top of the bounded delta parses.
			expect(parseCalls.streaming).toBeLessThan(1000);
			const deltas = events.filter((event) => event.type === "toolcall_delta").map((event) => event.delta);
			expect(deltas.join("")).toBe(json);
		});

		it("still parses every delta of a small tool call", async () => {
			const { json, chunks } = bigArguments(1);
			const { output } = await runResponses(chunks, json);

			expect((output.content[0] as ToolCall).arguments).toEqual(JSON.parse(json));
			// One parse per delta, plus one at arguments.done.
			expect(parseCalls.streaming).toBe(chunks.length + 1);
		});
	});
});
