import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <main className="runtime-error" role="alert">
          <p className="eyebrow">PhishGuard XAI</p>
          <h1>The research interface could not load.</h1>
          <p>{this.state.error.message || "Unknown client error"}</p>
          <button type="button" onClick={() => window.location.reload()}>
            Reload interface
          </button>
        </main>
      );
    }

    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </StrictMode>
);
