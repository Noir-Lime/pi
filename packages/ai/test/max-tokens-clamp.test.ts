import { describe, expect, it } from "vitest";
import { clampMaxTokensToContext } from "../src/api/simple-options.ts";
import type { Api, Model } from "../src/types.ts";
import { estimateContextTokens } from "../src/utils/estimate.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const model = { contextWindow: 500_000 } as Model<Api>;

// One user message of 3.5 characters per estimated token.
function contextOfTokens(tokens: number) {
	return normalizeContext({ messages: [{ role: "user", content: "x".repeat(tokens * 3.5), timestamp: 1 }] });
}

describe("clampMaxTokensToContext", () => {
	it("estimates the test contexts at the intended size", () => {
		expect(estimateContextTokens(contextOfTokens(400_000)).tokens).toBe(400_000);
	});

	it("keeps the requested ceiling instead of max_tokens=1 when the estimate leaves no room", () => {
		expect(clampMaxTokensToContext(model, contextOfTokens(8_000_000), 128_000)).toBe(128_000);
		expect(clampMaxTokensToContext(model, contextOfTokens(500_000), 128_000)).toBe(128_000);
		// available is exactly 0 here: 500_000 - 495_904 - 4_096
		expect(clampMaxTokensToContext(model, contextOfTokens(495_904), 128_000)).toBe(128_000);
	});

	it("still clamps when there is real but limited room, and leaves normal calls unchanged", () => {
		// available = 500_000 - 400_000 - 4_096
		expect(clampMaxTokensToContext(model, contextOfTokens(400_000), 128_000)).toBe(95_904);
		expect(clampMaxTokensToContext(model, contextOfTokens(495_903), 128_000)).toBe(1);
		expect(clampMaxTokensToContext(model, contextOfTokens(100_000), 8_192)).toBe(8_192);
		expect(clampMaxTokensToContext({ contextWindow: 0 } as Model<Api>, contextOfTokens(1), 5)).toBe(5);
	});
});
