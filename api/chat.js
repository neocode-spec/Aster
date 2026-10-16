
import { GoogleGenAI } from "@google/genai";

const INSTRUCTIONS = {
  "Dew Standard": `
You are Dew, the AI assistant inside Aster.
Answer clearly, naturally, and accurately.
Help with everyday questions, writing, coding, and problem-solving.
Be honest when you are uncertain.
Do not invent sources or claim to have searched when you have not.
`,

  "Dew Extended": `
You are Dew Extended, the advanced mode inside Aster.
Think carefully about complex questions.
Give detailed, structured answers, verify important claims with web search,
compare reliable sources, and explain uncertainty.
`,

  "Dew Search": `
You are Dew Search, the web research mode inside Aster.
Use Google Search to research the user's question.
Synthesize reliable information, cite relevant sources, and explain uncertainty.
`
};

const THINKING_BUDGET = {
  Low: 1024,
  Medium: 4096,
  High: 8192
};

const CURRENT_INFO_PATTERN =
  /\b(latest|current|currently|today|yesterday|recent|news|weather|price|pricing|2025|2026|2027|search|research|sources|release date|who won|this week)\b/i;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");

    return res.status(405).json({
      error: "Please send a POST request."
    });
  }

  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    return res.status(503).json({
      error: "Gemini API key is missing. Configure GEMINI_API_KEY in Vercel."
    });
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
      return res.status(400).json({
        error: "Please enter a message."
      });
    }

    if (message.length > 1200) {
      return res.status(400).json({
        error: "Messages must be 1,200 characters or fewer."
      });
    }

    const useSearch =
      mode === "Dew Search" ||
      mode === "Dew Extended" ||
      (
        mode === "Dew Standard" &&
        CURRENT_INFO_PATTERN.test(message)
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
      model: process.env.GEMINI_MODEL || "gemini-2.5-flash",
      contents: message,
      config
    });

    const answer = result.text?.trim();

    if (!answer) {
      return res.status(502).json({
        error: "Gemini returned an empty response. Please try again."
      });
    }

    const grounding =
      result.candidates?.[0]?.groundingMetadata;

    const sources = (grounding?.groundingChunks || [])
      .map(chunk => chunk.web)
      .filter(
        web =>
          web?.uri &&
          /^https?:\/\//i.test(web.uri)
      )
      .map(web => ({
        title: web.title || web.uri,
        url: web.uri
      }))
      .filter(
        (source, index, all) =>
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
    console.error(
      "Aster Gemini API error:",
      error?.message || "Unknown error"
    );

    return res.status(500).json({
      error: "Dew could not complete your request. Please try again."
    });
  }
}
