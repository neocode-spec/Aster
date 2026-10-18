
import { GoogleGenAI } from "@google/genai";
import { initializeApp, cert, getApps } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";
import {
  getFirestore,
  FieldValue,
  Timestamp
} from "firebase-admin/firestore";
import { createHash } from "node:crypto";

const MODE_INSTRUCTIONS = {
  "Dew Standard": `
You are Dew, the AI assistant inside Aster.
Help with everyday conversation, writing, coding, explanations,
research when appropriate, and problem-solving.
Be direct, accurate, useful, and reasonably concise.
  `,

  "Dew Extended": `
You are Dew Extended, the reasoning-focused assistant inside Aster.
Carefully analyze complex problems, verify calculations, and explain
the important steps in a clear, structured answer.
Do not expose private internal reasoning. Give useful explanations
and conclusions rather than an unfiltered reasoning transcript.
  `,

  "Dew Search": `
You are Dew Search, the research assistant inside Aster.
Use Google Search to research the user's request and ground factual
answers in current information whenever appropriate.
Prefer reliable and official sources. Distinguish verified facts
from uncertainty, and include useful source links when available.
Do not invent search results or citations.
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

const THINKING_BUDGETS = {
  Low: 1024,
  Medium: 2048,
  High: 4096,
  low: 1024,
  medium: 2048,
  high: 4096
};

const MAX_MESSAGE_LENGTH = 12000;
const HISTORY_LIMIT = 20;
const MEMORY_LIMIT = 10;

function getFirebase() {
  if (!getApps().length) {
    const raw = process.env.FIREBASE_SERVICE_ACCOUNT;

    if (!raw) {
      throw new Error("FIREBASE_SERVICE_ACCOUNT is missing");
    }

    const serviceAccount = JSON.parse(raw);

    initializeApp({
      credential: cert(serviceAccount),
      projectId: serviceAccount.project_id
    });
  }

  return {
    auth: getAuth(),
    db: getFirestore()
  };
}

function sendError(res, status, error) {
  return res.status(status).json({ error });
}

function normalizeRole(role) {
  if (role === "assistant" || role === "model") {
    return "model";
  }

  if (role === "user") {
    return "user";
  }

  return null;
}

// Merge consecutive messages with the same role instead of
// silently discarding their content.
function normalizeHistory(documents) {
  const history = [];

  for (const doc of documents) {
    const data = doc.data();
    const role = normalizeRole(data.role);
    const content =
      typeof data.content === "string" ? data.content.trim() : "";

    if (!role || !content) continue;

    const previous = history[history.length - 1];

    if (previous && previous.role === role) {
      previous.parts[0].text += "\n\n" + content;
    } else {
      history.push({
        role,
        parts: [{ text: content }]
      });
    }
  }

  // Gemini conversation history must begin with a user message.
  while (history.length && history[0].role !== "user") {
    history.shift();
  }

  return history;
}

function getModelName(mode) {
  if (mode === "Dew Extended") {
    return (
      process.env.GEMINI_EXTENDED_MODEL ||
      "gemini-2.5-pro"
    );
  }

  if (mode === "Dew Search") {
    return (
      process.env.GEMINI_SEARCH_MODEL ||
      process.env.GEMINI_MODEL ||
      "gemini-2.5-flash"
    );
  }

  return (
    process.env.GEMINI_STANDARD_MODEL ||
    process.env.GEMINI_MODEL ||
    "gemini-2.5-flash"
  );
}

function extractSources(result) {
  const candidate = result.candidates?.[0];
  const grounding = candidate?.groundingMetadata;

  const sourceMap = new Map();

  for (const chunk of grounding?.groundingChunks || []) {
    const web = chunk?.web;

    if (!web?.uri || !/^https?:\/\//i.test(web.uri)) {
      continue;
    }

    if (!sourceMap.has(web.uri)) {
      sourceMap.set(web.uri, {
        title: web.title || web.uri,
        url: web.uri
      });
    }
  }

  return {
    sources: Array.from(sourceMap.values()),
    searchUsed: Boolean(
      grounding?.webSearchQueries?.length ||
      grounding?.groundingChunks?.some(chunk => chunk?.web?.uri)
    )
  };
}

function buildSystemInstruction(mode, memoryContext) {
  return `
${MODE_INSTRUCTIONS[mode]}

MATHEMATICS AND FORMATTING:
- Preserve mathematical expressions accurately.
- For inline mathematics, use \\( ... \\).
- For displayed equations, use \\[ ... \\].
- Use standard LaTeX commands such as \\frac{a}{b},
  \\sqrt{x}, \\sum_{i=1}^{n}, \\begin{aligned}, and \\boxed{}.
- Put multi-line equations inside appropriate display-math delimiters.
- Do not place all mathematical expressions inside code blocks.
- Never claim that an equation was rendered visually by the frontend.
- Do not remove or escape LaTeX delimiters unnecessarily.

CODE:
- Use fenced code blocks for code.
- Preserve indentation and syntax.
- Explain important changes when debugging code.

${memoryContext}
  `.trim();
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return sendError(res, 405, "INVALID_METHOD");
  }

  try {
    // 1. Authenticate the Firebase user.
    const authorization = req.headers.authorization || "";
    const match = authorization.match(/^Bearer\s+(.+)$/i);

    if (!match) {
      return sendError(res, 401, "AUTH_REQUIRED");
    }

    const { auth, db } = getFirebase();

    let decoded;

    try {
      decoded = await auth.verifyIdToken(match[1]);
    } catch {
      return sendError(res, 401, "INVALID_AUTH_TOKEN");
    }

    const uid = decoded.uid;

    // 2. Validate the request.
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

    if (message.length > MAX_MESSAGE_LENGTH) {
      return sendError(res, 400, "MESSAGE_TOO_LONG");
    }

    const requestedMode = MODE_ALIASES[body.mode];
    const mode = requestedMode || "Dew Standard";

    const requestedThinking =
      body.reasoning ?? body.thinking ?? "Medium";

    const thinkingBudget =
      THINKING_BUDGETS[requestedThinking] ?? 2048;

    // Match the conversation ID used by the frontend and /api/chats.
    const requestedConversationId =
      typeof body.conversationId === "string"
        ? body.conversationId
        : "";

    const conversationId =
      /^[a-zA-Z0-9_-]{1,100}$/.test(requestedConversationId)
        ? requestedConversationId
        : "main";

    const userRef = db.collection("users").doc(uid);

    const conversationRef = userRef
      .collection("conversations")
      .doc(conversationId);

    const messagesRef = conversationRef.collection("messages");
    const memoriesRef = userRef.collection("memories");

    await userRef.set(
      { lastActiveAt: FieldValue.serverTimestamp() },
      { merge: true }
    );

    // 3. Load the most recent saved conversation messages.
    let history = [];

    try {
      const snapshot = await messagesRef
        .orderBy("createdAt", "desc")
        .limit(HISTORY_LIMIT)
        .get();

      const orderedDocuments = snapshot.docs.reverse();

      history = normalizeHistory(orderedDocuments);
    } catch (error) {
      console.error("Conversation history load failed:", error);
      return sendError(res, 500, "HISTORY_LOAD_FAILED");
    }

    // Add the current user message to the model context.
    const previous = history[history.length - 1];

    if (previous?.role === "user") {
      previous.parts[0].text += "\n\n" + message;
    } else {
      history.push({
        role: "user",
        parts: [{ text: message }]
      });
    }

    // 4. Load saved user memories.
    let memoryContext = "";

    try {
      const memorySnapshot = await memoriesRef
        .orderBy("updatedAt", "desc")
        .limit(MEMORY_LIMIT)
        .get();

      const memoryTexts = memorySnapshot.docs
        .map(doc => doc.data())
        .filter(item =>
          typeof item.content === "string" &&
          item.content.trim().length > 0
        )
        .map(item => {
          const category =
            typeof item.category === "string"
              ? item.category
              : "memory";

          return `- [${category}] ${item.content}`;
        });

      if (memoryTexts.length) {
        memoryContext =
          "\n\nSaved user memories:\n" + memoryTexts.join("\n");
      }
    } catch (error) {
      // Memory is optional. Continue without it if unavailable.
      console.warn("Memory load failed:", error);
    }

    // 5. Configure Gemini.
    if (!process.env.GEMINI_API_KEY) {
      console.error("GEMINI_API_KEY is missing");
      return sendError(res, 500, "AI_CONFIGURATION_ERROR");
    }

    const ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY
    });

    const config = {
      systemInstruction: buildSystemInstruction(
        mode,
        memoryContext
      )
    };

    if (mode === "Dew Search") {
      // Google Search is enabled for this mode.
      config.tools = [{ googleSearch: {} }];
    } else if (mode === "Dew Extended") {
      config.thinkingConfig = {
        thinkingBudget
      };
    } else {
      config.thinkingConfig = {
        thinkingBudget: 1024
      };
    }

    const model = getModelName(mode);

    // 6. Generate the response.
    const result = await ai.models.generateContent({
      model,
      contents: history,
      config
    });

    const answer =
      typeof result.text === "string"
        ? result.text.trim()
        : "";

    if (!answer) {
      console.error(
        "Gemini returned no text. Finish reason:",
        result.candidates?.[0]?.finishReason
      );

      return sendError(res, 502, "EMPTY_MODEL_RESPONSE");
    }

    // 7. Save the successful exchange to this conversation.
    // Distinct timestamps help maintain user-before-assistant ordering.
    const now = Date.now();

    const batch = db.batch();

    batch.set(
      messagesRef.doc(),
      {
        role: "user",
        content: message,
        createdAt: Timestamp.fromMillis(now)
      }
    );

    batch.set(
      messagesRef.doc(),
      {
        role: "assistant",
        content: answer,
        createdAt: Timestamp.fromMillis(now + 1)
      }
    );

    batch.set(
      conversationRef,
      {
        updatedAt: FieldValue.serverTimestamp(),
        title: FieldValue.delete(),
        lastMode: mode
      },
      { merge: true }
    );

    await batch.commit();

    // 8. Extract Google Search sources, if present.
    const { sources, searchUsed } = extractSources(result);

    // 9. Extract a small number of durable, useful memories.
    // Memory extraction must never cause a successful chat to fail.
    try {
      const extraction = await ai.models.generateContent({
        model:
          process.env.GEMINI_MEMORY_MODEL ||
          process.env.GEMINI_MODEL ||
          "gemini-2.5-flash",

        contents: [{
          role: "user",
          parts: [{
            text: `
Extract only durable, useful facts explicitly stated by the user
in the message below. Keep preferences, ongoing projects, goals,
and workflows. Do not infer sensitive personal facts.

Return valid JSON in this format:
{"memories":[{"category":"preference|project|goal","content":"fact"}]}

Return {"memories":[]} if there is nothing worth remembering.
Return no more than 2 memories.

User message:
${JSON.stringify(message)}
            `.trim()
          }]
        }],

        config: {
          responseMimeType: "application/json"
        }
      });

      const parsed = JSON.parse(
        extraction.text || '{"memories":[]}'
      );

      if (Array.isArray(parsed.memories)) {
        for (const item of parsed.memories.slice(0, 2)) {
          if (
            typeof item?.content !== "string" ||
            !item.content.trim()
          ) {
            continue;
          }

          const content = item.content.trim();

          const id = createHash("sha256")
            .update(content.toLowerCase())
            .digest("hex")
            .slice(0, 24);

          const allowedCategories = [
            "preference",
            "project",
            "goal"
          ];

          const category = allowedCategories.includes(item.category)
            ? item.category
            : "preference";

          await memoriesRef.doc(id).set(
            {
              category,
              content,
              updatedAt: FieldValue.serverTimestamp()
            },
            { merge: true }
          );
        }
      }
    } catch (error) {
      console.warn("Memory extraction failed:", error);
    }

    // 10. Return fields the frontend can consume.
    return res.status(200).json({
      answer,
      reply: answer,
      mode,
      thinking: requestedThinking,
      conversationId,
      searchUsed,
      sources
    });

  } catch (error) {
    console.error("Aster chat request failed:", error);

    if (error?.status === 429 || error?.status === 503) {
      return sendError(res, 503, "TEMPORARILY_UNAVAILABLE");
    }

    return sendError(res, 500, "REQUEST_FAILED");
  }
}
