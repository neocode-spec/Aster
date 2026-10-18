import { initializeApp } from "https://www.gstatic.com/firebasejs/11.10.0/firebase-app.js";
import {
  initializeAppCheck,
  ReCaptchaEnterpriseProvider
} from "https://www.gstatic.com/firebasejs/11.10.0/firebase-app-check.js";
import {
  getAI,
  GoogleAIBackend,
  getGenerativeModel
} from "https://www.gstatic.com/firebasejs/11.10.0/firebase-ai.js";
import {
  getAuth,
  GoogleAuthProvider
} from "https://www.gstatic.com/firebasejs/11.10.0/firebase-auth.js";

const firebaseConfig = {
  apiKey: "AIzaSyAhhJeujb2QvPsKnWheaDUpw0mtrdj6hkU",
  authDomain: "aster-20dew.firebaseapp.com",
  projectId: "aster-20dew",
  appId: "1:832735777031:web:89a1f2b7c47274e31b35a2"
};

const app = initializeApp(firebaseConfig);

// Protect production requests with App Check.
initializeAppCheck(app, {
  provider: new ReCaptchaEnterpriseProvider(
    "6Lc6L_ItAAAAAIfiNbKABSjfazHCVYxTPrJ0Pw27"
  ),
  isTokenAutoRefreshEnabled: true
});

// Firebase Authentication.
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

// Firebase AI Logic.
const ai = getAI(app, {
  backend: new GoogleAIBackend()
});

export const dewModel = getGenerativeModel(ai, {
  model: "gemini-3.8-flash"
});
