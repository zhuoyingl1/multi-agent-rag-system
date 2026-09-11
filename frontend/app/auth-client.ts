export type AuthenticatedUser = {
  user_id: string;
  email: string;
  display_name: string;
};

export type AuthSession = {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: AuthenticatedUser;
};

export type AuthCredentials = {
  email: string;
  password: string;
  display_name?: string;
};

const storageKey = "multi-agent-rag.auth-session";

export async function authenticate(
  apiBaseUrl: string,
  mode: "login" | "register",
  credentials: AuthCredentials,
): Promise<AuthSession> {
  const endpoint = mode === "login" ? "/auth/token" : "/auth/register";
  const response = await fetch(`${apiBaseUrl}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(credentials),
  });
  if (!response.ok) {
    throw new Error(await authErrorMessage(response));
  }
  return (await response.json()) as AuthSession;
}

export async function validateSession(apiBaseUrl: string, session: AuthSession): Promise<AuthSession> {
  const response = await authorizedFetch(`${apiBaseUrl}/auth/me`, session.access_token);
  if (!response.ok) {
    throw new Error("Your session is no longer valid.");
  }
  return { ...session, user: (await response.json()) as AuthenticatedUser };
}

export function authorizedFetch(input: RequestInfo | URL, token: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return fetch(input, { ...init, headers });
}

export function loadStoredSession(): AuthSession | null {
  if (typeof window === "undefined") {
    return null;
  }
  const value = window.sessionStorage.getItem(storageKey);
  if (!value) {
    return null;
  }
  try {
    return JSON.parse(value) as AuthSession;
  } catch {
    clearStoredSession();
    return null;
  }
}

export function storeSession(session: AuthSession) {
  window.sessionStorage.setItem(storageKey, JSON.stringify(session));
}

export function clearStoredSession() {
  if (typeof window !== "undefined") {
    window.sessionStorage.removeItem(storageKey);
  }
}

async function authErrorMessage(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail || `Authentication failed with ${response.status}`;
  } catch {
    return `Authentication failed with ${response.status}`;
  }
}
