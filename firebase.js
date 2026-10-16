
import { initializeApp } from "firebase/app";
import { initializeAppCheck, ReCaptchaEnterpriseProvider } from "firebase/app-check";
import { getAI, GoogleAIBackend, getGenerativeModel } from "firebase/ai";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

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
