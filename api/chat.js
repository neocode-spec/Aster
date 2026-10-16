
import { GoogleGenAI } from "@google/genai";

const INSTRUCTIONS = {
  "Dew Standard": `
You are Dew, the AI assistant inside Aster.
Handle greetings, everyday conversation, explanations, writing,
coding, and problem-solving naturally using your existing knowledge.
Do not use web search unless the user explicitly requests it.
Be honest about uncertainty and never invent sources.
`,

  "Dew Extended": `
You are Dew Extended, the reasoning-focused mode inside Aster.
Think carefully about complex questions and give detailed,
well-structured answers. Use your existing knowledge by default.
Do not use web search unless the user explicitly requests it.
Be honest about uncertainty and never invent sources.
`,

  "Dew Search": `
You are Dew Search, the web research mode inside Aster.
Use Google Search to research the user's question.
Synthesize reliable information, cite relevant sources,
and explain uncertainty. Never claim a search succeeded if it did not.
`
};

const THINKING_BUDGET = {
  Low: 1024,
  Medium: 4096,
  High: 8192
};

// Search in Standard or Extended only when explicitly requested.
const EXPLICIT_SEARCH_PATTERN =
  /\b(search (the )?web|search online|look (it|this|that|them) up|look online|google (it|this|that)|browse (the )?web|check online|find (online|on the web)|research (this|that|the topic|online)|use web search|look for sources online)\b/i;

function sendError(res, status, code) {
  return res.status(status).json({ error: code });
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

    const mode = Object.hasOwn(INSTRUCTIONS, body.mode)
      ? body.mode
      : "Dew Standard";

    const thinking = ["Low", "Medium", "High"].includes(body.thinking)
      ? body.thinking
      : "Medium";

    if (!message) {
      return sendError(res, 400, "EMPTY_MESSAGE");
    }

    if (message.length > 1200) {
      return sendError(res, 400, "MESSAGE_TOO_LONG");
    }

    const explicitlyRequestedSearch =
      EXPLICIT_SEARCH_PATTERN.test(message);

    // Standard never searches automatically.
    // Extended never searches automatically.
    // Dew Search always attempts web search.
    const useSearch =
      mode === "Dew Search" ||
      (
        (mode === "Dew Standard" || mode === "Dew Extended") &&
        explicitlyRequestedSearch
      );

    const ai = new GoogleGenAI({ apiKey });

    const config = {
      systemInstruction: INSTRUCTIONS[mode],
      thinkingConfig: {
        thinkingBudget: THINKING_BUDGET[thinking]
      }
    };

    if (useSearch) {
      config.tools = [{ googleSearch: {} }];
    }

    const result = await ai.models.generateContent({
      model: process.env.GEMINI_MODEL || "gemini-3.8-flash",
      contents: message,
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

    return res.status(200).json({
      answer,
      mode,
      thinking,
      searchUsed: Boolean(grounding),
      sources
    });
  } catch (error) {
    // Keep technical details in server logs, not the public chat.
    console.error("Aster Gemini request failed:", error);

    const status = Number(error?.status || error?.code);

    if (
      status === 503 ||
      status === 429 ||
      status === 502 ||
      status === 504
    ) {
      return sendError(
        res,
        503,
        "TEMPORARILY_UNAVAILABLE"
      );
    }

    if (status === 401 || status === 403) {
      return sendError(res, 503, "SERVICE_UNAVAILABLE");
    }

    return sendError(res, 500, "REQUEST_FAILED");
  }
}
