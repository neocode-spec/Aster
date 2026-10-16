
import { GoogleGenAI } from "@google/genai";

const INSTRUCTIONS = {
  "Dew Standard": `
You are Dew, the AI assistant inside Aster.
Handle greetings, everyday conversation, explanations, writing,
coding, and problem-solving naturally using your existing knowledge.
Never activate web search automatically.
Be honest about uncertainty and never invent sources.
`,

  "Dew Extended": `
You are Dew Extended, the reasoning-focused mode inside Aster.
Think carefully about complex questions and provide detailed,
well-structured answers.
Use your existing knowledge. Never activate web search automatically.
Be honest about uncertainty and never invent sources.
`,

  "Dew Search": `
You are Dew Search, the web research mode inside Aster.
Use Google Search to research the user's question.
Synthesize reliable information, cite relevant sources when available,
and explain uncertainty. Never claim a search succeeded if it did not.
`
};

const MODE_ALIASES = {
  standard: "Dew Standard",
  extended: "Dew Extended",
  search: "Dew Search",
  "Dew Standard": "Dew Standard",
  "Dew Extended": "Dew Extended",
  "Dew Search": "Dew Search"
};

const THINKING_BUDGET = {
  Low: 1024,
  Medium: 4096,
  High: 8192
};

const THINKING_ALIASES = {
  low: "Low",
  medium: "Medium",
  high: "High",
  Low: "Low",
  Medium: "Medium",
  High: "High"
};

function sendError(res, status, code) {
  return res.status(status).json({ error: code });
}

function normalizeHistory(history, message) {
  const validHistory = Array.isArray(history)
    ? history
        .filter(item =>
          item &&
          ["user", "assistant", "model"].includes(item.role) &&
          typeof item.content === "string" &&
          item.content.trim()
        )
        .slice(-20)
    : [];

  const contents = validHistory.map(item => ({
    role: item.role === "user" ? "user" : "model",
    parts: [{ text: item.content.trim() }]
  }));

  const last = contents[contents.length - 1];
  const lastText = last?.parts?.[0]?.text;

  // The frontend history normally already contains the latest message.
  // Append it only when it is not already the final user message.
  if (
    !last ||
    last.role !== "user" ||
    lastText !== message
  ) {
    contents.push({
      role: "user",
      parts: [{ text: message }]
    });
  }

  return contents;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return sendError(res, 405, "INVALID_METHOD");
  }

  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    console.error("Aster configuration error: GEMINI_API_KEY is missing.");
    return sendError(res, 503, "SERVICE_UNAVAILABLE");
  }

  try {
    const body =
      typeof req.body === "string"
        ? JSON.parse(req.body)
        : req.body || {};

    const message =
      typeof body.message === "string"
        ? body.message.trim()
        : "";

    if (!message) {
      return sendError(res, 400, "EMPTY_MESSAGE");
    }

    if (message.length > 12000) {
      return sendError(res, 400, "MESSAGE_TOO_LONG");
    }

    // Accept both the existing backend labels and the new frontend values.
    const mode = MODE_ALIASES[body.mode] || "Dew Standard";

    // The frontend sends "reasoning"; older versions may send "thinking".
    const thinkingInput = body.reasoning ?? body.thinking ?? "Medium";
    const thinking =
      THINKING_ALIASES[thinkingInput] || "Medium";

    const ai = new GoogleGenAI({ apiKey });

    const config = {
      systemInstruction: INSTRUCTIONS[mode],
      thinkingConfig: {
        thinkingBudget: THINKING_BUDGET[thinking]
      }
    };

    // Only the explicitly selected Dew Search mode activates Google Search.
    const useSearch = mode === "Dew Search";

    if (useSearch) {
      config.tools = [{ googleSearch: {} }];
    }

    const contents = normalizeHistory(body.history, message);

    const result = await ai.models.generateContent({
      model: process.env.GEMINI_MODEL || "gemini-3.8-flash",
      contents,
      config
    });

    const answer = result.text?.trim();

    if (!answer) {
      console.error("Aster received an empty model response.");
      return sendError(res, 502, "EMPTY_MODEL_RESPONSE");
    }

    const grounding = result.candidates?.[0]?.groundingMetadata;

    const sources = (grounding?.groundingChunks || [])
      .map(chunk => chunk.web)
      .filter(web => web?.uri && /^https?:\/\//i.test(web.uri))
      .map(web => ({
        title: web.title || web.uri,
        url: web.uri
      }))
      .filter((source, index, all) =>
        all.findIndex(item => item.url === source.url) === index
      );

    const searchUsed =
      useSearch && Boolean(grounding?.webSearchQueries?.length);

    return res.status(200).json({
      answer,
      reply: answer,
      mode,
      thinking,
      reasoning: thinking,
      searchUsed,
      sources
    });

  } catch (error) {
    console.error("Aster Gemini request failed:", error);

    const status = Number(error?.status || error?.code);

    if (status === 429 || status === 503 || status === 502 || status === 504) {
      return sendError(res, 503, "TEMPORARILY_UNAVAILABLE");
    }

    if (status === 401 || status === 403) {
      return sendError(res, 503, "SERVICE_UNAVAILABLE");
    }

    if (error instanceof SyntaxError) {
      return sendError(res, 400, "INVALID_REQUEST");
    }

    return sendError(res, 500, "REQUEST_FAILED");
  }
}
