import { GoogleGenAI } from "@google/genai";
import { initializeApp, cert, getApps } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";
import { getFirestore, FieldValue } from "firebase-admin/firestore";
import { createHash } from "node:crypto";

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

const INSTRUCTIONS = {
  "Dew Standard": `You are Dew, the AI assistant inside Aster. Handle everyday conversation, writing, coding and problem-solving. Be concise, direct and helpful.`,
  "Dew Extended": `You are Dew Extended, the reasoning-focused assistant inside Aster. Reason step-by-step, verify your logic carefully, and provide thorough structured answers.`,
  "Dew Search": `You are Dew Search, the research mode inside Aster. Use real-time Google Search to provide up-to-date facts, current news, and verified sources.`
};

const MODES = {
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

function errorResponse(res, status, error, details) {
  return res.status(status).json({ error, details: details || undefined });
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return errorResponse(res, 405, "INVALID_METHOD");
  }

  try {
    const authorization = req.headers.authorization || "";
    const match = authorization.match(/^Bearer\s+(.+)$/i);
    if (!match) {
      return errorResponse(res, 401, "AUTH_REQUIRED");
    }

    const { auth, db } = getFirebase();
    const decoded = await auth.verifyIdToken(match[1]);
    const uid = decoded.uid;

    const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};
    const message = typeof body.message === "string" ? body.message.trim() : "";

    if (!message) return errorResponse(res, 400, "EMPTY_MESSAGE");
    if (message.length > 12000) return errorResponse(res, 400, "MESSAGE_TOO_LONG");

    const mode = MODES[body.mode] || "Dew Standard";
    const isSearch = (mode === "Dew Search");
    const isExtended = (mode === "Dew Extended");

    const userRef = db.collection("users").doc(uid);
    const conversationRef = userRef.collection("conversations").doc("main");
    const messagesRef = conversationRef.collection("messages");
    const memoriesRef = userRef.collection("memories");

    await userRef.set({ lastActiveAt: FieldValue.serverTimestamp() }, { merge: true });

    // Fetch conversation history
    let history = [];
    try {
      const historySnapshot = await messagesRef.orderBy("createdAt", "desc").limit(12).get();
      const rawHistory = historySnapshot.docs.reverse().map(doc => {
        const item = doc.data();
        return {
          role: item.role === "assistant" ? "model" : "user",
          parts: [{ text: String(item.content || "") }]
        };
      }).filter(h => h.parts[0].text.trim().length > 0);

      // Clean history to ensure strict alternating roles (user -> model -> user)
      for (const turn of rawHistory) {
        if (!history.length || history[history.length - 1].role !== turn.role) {
          history.push(turn);
        }
      }
    } catch (e) {
      console.warn("Could not load history, starting fresh:", e);
      history = [];
    }

    // Append the current message
    history.push({
      role: "user",
      parts: [{ text: message }]
    });

    // Ensure the history starts with a user turn
    while (history.length && history[0].role !== "user") {
      history.shift();
    }

    // Load user memories
    let memoryContext = "";
    try {
      const memorySnapshot = await memoriesRef.orderBy("updatedAt", "desc").limit(10).get();
      const memoryTexts = memorySnapshot.docs
        .map(doc => doc.data())
        .filter(item => typeof item?.content === "string")
        .map(item => `- [${item.category}] ${item.content}`);
      if (memoryTexts.length) {
        memoryContext = "\n\nRelevant user facts/memories:\n" + memoryTexts.join("\n");
      }
    } catch (e) {
      console.warn("Could not load memories:", e);
    }

    const ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY
    });

    // Build configuration cleanly per mode
    const config = {
      systemInstruction: INSTRUCTIONS[mode] + memoryContext
    };

    if (isSearch) {
      // GOOGLE SEARCH MODE: Enable web search tool. No thinkingBudget to prevent collisions!
      config.tools = [{ googleSearch: {} }];
    } else if (isExtended) {
      // EXTENDED MODE: Deep reasoning budget
      const budget = THINKING_BUDGETS[body.reasoning ?? body.thinking ?? "Medium"] || 2048;
      config.thinkingConfig = { thinkingBudget: budget };
    } else {
      // STANDARD MODE: Fast everyday reasoning
      config.thinkingConfig = { thinkingBudget: 1024 };
    }

    // Official verified models: gemini-2.5-flash for fast/search, gemini-2.5-pro for extended
    const model = isExtended ? "gemini-2.5-pro" : "gemini-2.5-flash";

    const result = await ai.models.generateContent({
      model: process.env.GEMINI_MODEL || model,
      contents: history,
      config
    });

    const answer = result.text?.trim();
    if (!answer) {
      return errorResponse(res, 502, "EMPTY_MODEL_RESPONSE");
    }

    // Save user message and model answer into Firestore
    await messagesRef.add({
      role: "user",
      content: message,
      createdAt: FieldValue.serverTimestamp()
    });

    await messagesRef.add({
      role: "assistant",
      content: answer,
      createdAt: FieldValue.serverTimestamp()
    });

    await conversationRef.set({ updatedAt: FieldValue.serverTimestamp() }, { merge: true });

    // Safely extract grounding search sources
    let sources = [];
    let searchUsed = false;
    try {
      const candidate = result.candidates?.[0];
      const grounding = candidate?.groundingMetadata;
      if (grounding) {
        searchUsed = Boolean(grounding.webSearchQueries?.length);
        if (Array.isArray(grounding.groundingChunks)) {
          sources = grounding.groundingChunks
            .map(chunk => chunk?.web)
            .filter(web => web?.uri && /^https?:\/\//i.test(web.uri))
            .map(web => ({
              title: web.title || web.uri,
              url: web.uri
            }))
            .filter((source, index, all) => all.findIndex(item => item.url === source.url) === index);
        }
      }
    } catch (e) {
      console.warn("Grounding extraction skipped:", e);
    }

    // Background memory extraction
    try {
      const extraction = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [{
          role: "user",
          parts: [{
            text: `Extract useful user facts from this message:\n"${message}"\nReturn JSON: {"memories":[{"category":"preference|project|goal","content":"fact"}]}. Return {"memories":[]} if none.`
          }]
        }],
        config: { responseMimeType: "application/json" }
      });
      const parsed = JSON.parse(extraction.text || '{"memories":[]}');
      if (Array.isArray(parsed.memories)) {
        for (const item of parsed.memories.slice(0, 2)) {
          if (item?.content && typeof item.content === "string") {
            const id = createHash("sha256").update(item.content.toLowerCase()).digest("hex").slice(0, 24);
            await memoriesRef.doc(id).set({
              category: item.category || "preference",
              content: item.content.trim(),
              updatedAt: FieldValue.serverTimestamp()
            }, { merge: true });
          }
        }
      }
    } catch (e) {
      // Non-blocking
    }

    return res.status(200).json({
      answer,
      reply: answer,
      mode,
      searchUsed,
      sources
    });

  } catch (error) {
    console.error("Aster chat request failed:", error);

    if (error?.code === "auth/argument-error" ||
        error?.code === "auth/id-token-expired" ||
        error?.code === "auth/invalid-id-token") {
      return errorResponse(res, 401, "INVALID_AUTH_TOKEN");
    }

    if (error?.status === 429 || error?.status === 503) {
      return errorResponse(res, 503, "TEMPORARILY_UNAVAILABLE");
    }

    return errorResponse(res, 500, "REQUEST_FAILED", error?.message);
  }
}
