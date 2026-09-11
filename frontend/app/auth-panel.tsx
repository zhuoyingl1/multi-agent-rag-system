"use client";

import { Loader2, LockKeyhole, LogIn, UserPlus, X } from "lucide-react";
import { FormEvent, useState } from "react";

import { AuthSession, authenticate } from "./auth-client";

type AuthPanelProps = {
  apiBaseUrl: string;
  onAuthenticated: (session: AuthSession) => void;
  onClose?: () => void;
};

export default function AuthPanel({ apiBaseUrl, onAuthenticated, onClose }: AuthPanelProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      const session = await authenticate(apiBaseUrl, mode, {
        email: email.trim(),
        password,
        ...(mode === "register" ? { display_name: displayName.trim() } : {}),
      });
      onAuthenticated(session);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Authentication failed.");
    } finally {
      setSubmitting(false);
    }
  }

  function changeMode(nextMode: "login" | "register") {
    setMode(nextMode);
    setError("");
  }

  return (
    <section className="authPanel" aria-label="Account access">
      <div className="authPanelHeader">
        <LockKeyhole size={21} />
        <h2>{mode === "login" ? "Sign in" : "Create account"}</h2>
        {onClose && (
          <button className="authCloseButton" type="button" onClick={onClose} aria-label="Close" title="Close">
            <X size={18} />
          </button>
        )}
      </div>

      <div className="authModeSwitch" aria-label="Authentication mode">
        <button type="button" className={mode === "login" ? "active" : ""} onClick={() => changeMode("login")}>
          <LogIn size={16} />
          Sign in
        </button>
        <button
          type="button"
          className={mode === "register" ? "active" : ""}
          onClick={() => changeMode("register")}
        >
          <UserPlus size={16} />
          Register
        </button>
      </div>

      <form className="authForm" onSubmit={submit}>
        {mode === "register" && (
          <>
            <label htmlFor="authDisplayName">Display name</label>
            <input
              id="authDisplayName"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              autoComplete="name"
              maxLength={100}
              required
            />
          </>
        )}

        <label htmlFor="authEmail">Email</label>
        <input
          id="authEmail"
          type="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          autoComplete="email"
          maxLength={254}
          required
        />

        <label htmlFor="authPassword">Password</label>
        <input
          id="authPassword"
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          minLength={8}
          maxLength={128}
          required
        />

        {error && <div className="authError" role="alert">{error}</div>}

        <button className="authSubmitButton" type="submit" disabled={submitting}>
          {submitting ? <Loader2 className="spin" size={17} /> : mode === "login" ? <LogIn size={17} /> : <UserPlus size={17} />}
          {mode === "login" ? "Sign in" : "Create account"}
        </button>
      </form>
    </section>
  );
}
