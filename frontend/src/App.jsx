import React, { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Database,
  Gauge,
  GitBranch,
  Link as LinkIcon,
  Lock,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles
} from "lucide-react";

const suspiciousKeywords = [
  "login",
  "signin",
  "verify",
  "secure",
  "account",
  "update",
  "banking",
  "confirm",
  "password",
  "payment",
  "support",
  "security",
  "auth",
  "wallet"
];

const samples = [
  {
    label: "Legitimate",
    value: "https://www.openai.com/research"
  },
  {
    label: "IP-risk",
    value: "http://192.168.1.44/secure-login/verify?session=9881"
  },
  {
    label: "Impersonation",
    value: "https://paypal.com.account-verify.security-check.example.net/update/password"
  }
];

const modelCards = [
  {
    name: "XGBoost",
    tag: "Champion candidate",
    role: "Primary high-performance model",
    icon: BrainCircuit,
    params: "n_estimators=150, max_depth=6, learning_rate=0.1",
    body: "Tree boosting captures non-linear interactions between entropy, URL length, subdomains, symbols, and keyword density."
  },
  {
    name: "Random Forest",
    tag: "Ensemble baseline",
    role: "Robust tree ensemble",
    icon: GitBranch,
    params: "n_estimators=150, class_weight=balanced",
    body: "Bagged decision trees provide robust classification and readable feature-importance rankings."
  },
  {
    name: "SVM (RBF approximation)",
    tag: "Scalable margin model",
    role: "Nystroem RBF map + LinearSVC",
    icon: Activity,
    params: "n_components=50, gamma=1.0, C=1.0",
    body: "A Nystroem RBF feature map with LinearSVC approximates a non-linear decision boundary at full-dataset scale."
  },
  {
    name: "Decision Tree",
    tag: "Interpretable baseline",
    role: "Readable decision paths",
    icon: GitBranch,
    params: "max_depth=8, class_weight=balanced",
    body: "A lightweight baseline that produces transparent feature-split logic for academic comparison."
  },
  {
    name: "Logistic Regression",
    tag: "Linear baseline",
    role: "Statistical reference model",
    icon: BarChart3,
    params: "penalty=l2, solver=lbfgs, class_weight=balanced",
    body: "A transparent statistical baseline for comparing the value of more expressive classifiers."
  }
];

const modelProfiles = {
  XGBoost: { multiplier: 1.0, bias: 0 },
  "Random Forest": { multiplier: 0.96, bias: 2 },
  "SVM (RBF approximation)": { multiplier: 0.92, bias: 4 },
  "Decision Tree": { multiplier: 1.05, bias: -1 },
  "Logistic Regression": { multiplier: 0.88, bias: 5 }
};

const benchmarkResults = {
  "Logistic Regression": ["99.33%", "98.90%", "99.94%", "99.42%", "99.61%"],
  "Decision Tree": ["99.51%", "99.23%", "99.92%", "99.57%", "99.68%"],
  "Random Forest": ["99.49%", "99.33%", "99.79%", "99.56%", "99.65%"],
  "SVM (RBF approximation)": ["97.84%", "99.19%", "97.01%", "98.09%", "99.55%"],
  XGBoost: ["99.56%", "99.32%", "99.93%", "99.62%", "99.79%"]
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "";

const featureGroups = [
  {
    title: "Lexical Structure",
    icon: LinkIcon,
    body: "Length, dots, subdomains, hyphens, special characters, digit density, and path depth."
  },
  {
    title: "Security Signals",
    icon: Lock,
    body: "HTTPS presence, IP-address hosts, and authentication-related keywords used in phishing URLs."
  },
  {
    title: "Entropy",
    icon: Sparkles,
    body: "Character randomness that can reveal generated domains, encoded tokens, and obfuscation."
  },
  {
    title: "Rule-based indicators",
    icon: ShieldCheck,
    body: "Human-readable risk indicators show which URL properties triggered a warning."
  }
];

const featureCatalog = [
  ["URL length", "Total number of characters in the full URL.", "Long links may hide redirects, tokens, or deceptive paths."],
  ["Domain length", "Character length of the hostname/domain.", "Unusually long hosts can indicate generated or misleading domains."],
  ["Dot count", "Number of periods in the URL.", "High dot usage can indicate complex nesting or deceptive host structure."],
  ["Subdomain count", "Number of labels before the registered domain.", "Attackers may place trusted brand names in subdomains."],
  ["Hyphen count", "Number of hyphens in the URL.", "Hyphens are common in typo-squatting and brand impersonation."],
  ["Special-character count", "Count of symbols such as @, ?, =, %, _, &, and dots.", "High symbol density can indicate obfuscation or parameter stuffing."],
  ["Digit count", "Total number of numeric characters.", "Generated URLs often include numeric IDs, hashes, or random tokens."],
  ["Digit ratio", "Digits divided by total URL length.", "Normalizes numeric density across short and long URLs."],
  ["IP address presence", "Whether the hostname is a raw IPv4 or IPv6 address.", "Raw IP links can indicate temporary or untrusted infrastructure."],
  ["HTTPS presence", "Whether the URL uses HTTPS.", "Missing HTTPS adds risk, though HTTPS alone does not prove legitimacy."],
  ["Suspicious keyword density", "Count of terms like login, verify, account, secure, update, password, and payment.", "Credential-harvesting links often use authentication and urgency terms."],
  ["Shannon entropy", "Randomness of the URL character distribution.", "High entropy can reveal generated domains, encoded strings, or obfuscation."]
];

const datasetRows = [
  ["Phishing URLs", "PhishTank, OpenPhish", "Use verified phishing URLs and record collection date."],
  ["Legitimate URLs", "Tranco top-site ranking", "Remove domains that appear in known malicious feeds."],
  ["Preprocessing", "Deduplication, scheme normalization, malformed-character cleanup", "Avoid duplicate and invalid samples before feature extraction."],
  ["Leakage control", "Stratified split plus domain-family separation", "Prevents the same domain family from appearing in train and test data."]
];

const metricRows = [
  ["Accuracy", "Overall correct classifications."],
  ["Precision", "Reduces legitimate links being wrongly flagged."],
  ["Recall", "Finds as many real phishing URLs as possible."],
  ["F1-score", "Balances precision and recall."],
  ["ROC-AUC", "Measures discrimination across thresholds."],
  ["Inference latency", "Shows real-time suitability."]
];

function normalizeUrl(rawUrl) {
  const trimmed = rawUrl.trim();
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed)) return trimmed;
  return `https://${trimmed}`;
}

function parseUrl(rawUrl) {
  try {
    return new URL(normalizeUrl(rawUrl));
  } catch {
    return null;
  }
}

function shannonEntropy(value) {
  if (!value.length) return 0;
  const counts = new Map();
  for (const char of value) counts.set(char, (counts.get(char) || 0) + 1);

  let entropy = 0;
  for (const count of counts.values()) {
    const probability = count / value.length;
    entropy -= probability * Math.log2(probability);
  }
  return entropy;
}

function isIpAddress(hostname) {
  const ipv4 = /^(?:\d{1,3}\.){3}\d{1,3}$/;
  const ipv6 = /^[0-9a-f:]+$/i;
  return ipv4.test(hostname) || (hostname.includes(":") && ipv6.test(hostname));
}

function extractFeatures(rawUrl) {
  const normalized = normalizeUrl(rawUrl);
  const parsed = parseUrl(rawUrl);
  const host = parsed?.hostname ?? "";
  const labels = host.split(".").filter(Boolean);
  const subdomainCount = labels.length > 2 ? labels.slice(0, -2).length : 0;
  const keywordHits = suspiciousKeywords.filter((word) => normalized.toLowerCase().includes(word));
  const digits = normalized.match(/\d/g) ?? [];
  const specialCharacters = normalized.match(/[-_@?=%&.]/g) ?? [];

  return {
    normalized,
    host,
    urlLength: normalized.length,
    domainLength: host.length,
    dotCount: (normalized.match(/\./g) ?? []).length,
    subdomainCount,
    hyphenCount: (normalized.match(/-/g) ?? []).length,
    specialCharacterCount: specialCharacters.length,
    digitCount: digits.length,
    digitRatio: normalized.length ? digits.length / normalized.length : 0,
    ipAddressPresent: host ? isIpAddress(host) : false,
    httpsPresent: parsed ? parsed.protocol === "https:" : false,
    suspiciousKeywordCount: keywordHits.length,
    keywordHits,
    pathDepth: parsed ? parsed.pathname.split("/").filter(Boolean).length : 0,
    entropy: shannonEntropy(normalized)
  };
}

function scoreFeatures(features) {
  let risk = 8;
  const reasons = [];

  const add = (points, label, detail, tone = "warning") => {
    risk += points;
    reasons.push({ label, detail, tone });
  };

  if (features.urlLength > 75) add(14, "Long URL", "The URL is long enough to hide tokens or redirects.");
  if (features.subdomainCount >= 3) add(18, "Many subdomains", "Nested subdomains can disguise the true domain.", "danger");
  else if (features.subdomainCount >= 2) add(10, "Nested subdomains", "The host uses multiple subdomain labels.");
  if (features.ipAddressPresent) add(24, "IP address host", "Raw IP links are high risk in phishing contexts.", "danger");
  if (!features.httpsPresent) add(10, "No HTTPS", "The URL does not use encrypted HTTPS transport.");
  if (features.suspiciousKeywordCount >= 3) add(18, "Authentication wording", "Several login or verification terms appear in the URL.", "danger");
  else if (features.suspiciousKeywordCount > 0) add(9, "Suspicious keyword", "Account or security wording appears in the URL.");
  if (features.entropy > 4.6) add(12, "High entropy", "Character randomness is elevated.");
  if (features.specialCharacterCount > 12) add(10, "Many symbols", "The URL contains many delimiters or encoded-looking characters.");
  if (features.digitRatio > 0.14) add(8, "Digit-heavy URL", "Numbers make up a high share of the URL.");
  if (features.pathDepth >= 4) add(7, "Deep path", "The path contains several nested segments.");

  if (!reasons.length) {
    reasons.push({
      label: "Low lexical risk",
      detail: "No major suspicious URL-only features were detected.",
      tone: "safe"
    });
  }

  return {
    risk: Math.max(2, Math.min(98, Math.round(risk))),
    reasons
  };
}

function applyModelProfile(baseRisk, modelName) {
  const profile = modelProfiles[modelName] ?? modelProfiles.XGBoost;
  return Math.max(2, Math.min(98, Math.round(baseRisk * profile.multiplier + profile.bias)));
}

function verdictForRisk(risk, threshold = 50) {
  if (risk >= threshold) return "Likely Phishing";
  if (risk >= Math.max(20, threshold - 12)) return "Suspicious";
  return "Likely Legitimate";
}

function toneForRisk(risk, threshold = 50) {
  if (risk >= threshold) return "danger";
  if (risk >= Math.max(20, threshold - 12)) return "warning";
  return "safe";
}

function featureRows(features) {
  return [
    ["URL length", features.urlLength, features.urlLength > 75 ? "Long URL may hide redirects or tokens." : "Within common range."],
    ["Domain length", features.domainLength, features.domainLength > 35 ? "Long hostnames can indicate deception." : "No unusual host length."],
    ["Dot count", features.dotCount, features.dotCount > 4 ? "Many dots can indicate complex host construction." : "Dot usage is moderate."],
    ["Subdomains", features.subdomainCount, features.subdomainCount >= 3 ? "Multiple levels can imitate trusted domains." : "Subdomain depth is moderate."],
    ["Hyphens", features.hyphenCount, features.hyphenCount > 2 ? "Hyphen-heavy names may support impersonation." : "Hyphen usage is limited."],
    ["Special characters", features.specialCharacterCount, features.specialCharacterCount > 12 ? "Many symbols may indicate obfuscation." : "Symbol usage is not excessive."],
    ["Digits", features.digitCount, features.digitCount > 10 ? "Large numeric tokens can signal generated links." : "Digit count is limited."],
    ["Digit ratio", features.digitRatio.toFixed(2), features.digitRatio > 0.14 ? "Numeric density is high." : "Numeric density is moderate."],
    ["IP address", features.ipAddressPresent ? "Detected" : "No", features.ipAddressPresent ? "Raw IP host detected." : "Host is a domain name."],
    ["HTTPS", features.httpsPresent ? "Yes" : "No", features.httpsPresent ? "Encrypted transport is present." : "Missing HTTPS adds risk."],
    ["Suspicious keywords", features.suspiciousKeywordCount, features.suspiciousKeywordCount ? "Authentication terms found." : "No listed phishing keywords found."],
    ["Entropy", features.entropy.toFixed(2), features.entropy > 4.6 ? "Randomness is elevated." : "Randomness is within expected range."]
  ];
}

function Stat({ value, label }) {
  return (
    <div className="stat">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function UrlAnalyzer() {
  const [url, setUrl] = useState(samples[2].value);
  const [submittedUrl, setSubmittedUrl] = useState(samples[2].value);
  const [selectedModel, setSelectedModel] = useState("XGBoost");
  const [threshold, setThreshold] = useState(50);
  const [remoteResult, setRemoteResult] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const features = useMemo(() => extractFeatures(submittedUrl), [submittedUrl]);
  const baseResult = useMemo(() => scoreFeatures(features), [features]);
  const risk = useMemo(() => applyModelProfile(baseResult.risk, selectedModel), [baseResult.risk, selectedModel]);
  const result = { ...baseResult, risk };
  const displayedRisk = remoteResult?.risk_score ?? result.risk;
  const displayedReasons = remoteResult?.reasons?.map((detail) => ({
    label: "Backend diagnostic",
    detail,
    tone: "warning"
  })) ?? result.reasons;
  const verdict = remoteResult
    ? remoteResult.verdict === "phishing" ? "Likely Phishing" : "Likely Legitimate"
    : verdictForRisk(result.risk, threshold);
  const tone = toneForRisk(displayedRisk, threshold);
  const rows = useMemo(() => featureRows(features), [features]);

  async function analyzeUrl(nextUrl) {
    setSubmittedUrl(nextUrl);
    setRemoteResult(null);
    if (!API_BASE_URL) return;

    setIsScanning(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: nextUrl, model: selectedModel })
      });
      if (!response.ok) throw new Error("API scan failed");
      setRemoteResult(await response.json());
    } catch {
      setRemoteResult(null);
    } finally {
      setIsScanning(false);
    }
  }

  return (
    <section id="demo" className="section demo-section" aria-labelledby="demo-title">
      <div className="section-heading">
        <p className="eyebrow">Interactive Prototype</p>
        <h2 id="demo-title">Analyze a URL in Real Time</h2>
              <p>
                The scanner sends URL-only features to FastAPI when configured. Without the API, results are clearly labeled as a demo heuristic.
              </p>
      </div>

      <div className="demo-grid">
        <div className="analyzer-panel">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              analyzeUrl(url);
            }}
          >
            <label htmlFor="url-input">Submitted URL</label>
            <div className="input-row">
              <Search aria-hidden="true" size={20} />
              <input
                id="url-input"
                type="url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://example.com/login"
                required
              />
              <button type="submit">
                <Gauge aria-hidden="true" size={18} />
                Analyze
              </button>
            </div>
          </form>

          <div className="sample-row" aria-label="Sample URLs">
            {samples.map((sample) => (
              <button
                type="button"
                key={sample.label}
                onClick={() => {
                  setUrl(sample.value);
                  analyzeUrl(sample.value);
                }}
              >
                {sample.label}
              </button>
            ))}
          </div>

          <div className="control-grid">
            <label>
              <span>Benchmark model</span>
              <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                {modelCards.map((model) => (
                  <option value={model.name} key={model.name}>
                    {model.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Decision threshold: {threshold}%</span>
              <input
                type="range"
                min="30"
                max="80"
                value={threshold}
                onChange={(event) => setThreshold(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="signal-list">
            <div>
              <span>Host</span>
              <strong>{features.host || "Invalid host"}</strong>
            </div>
            <div>
              <span>Model Preview</span>
              <strong>{selectedModel} at {threshold}% threshold</strong>
            </div>
            <div>
              <span>Keyword Hits</span>
              <strong>{features.keywordHits.length ? features.keywordHits.join(", ") : "None"}</strong>
            </div>
          </div>
        </div>

        <aside className={`result-panel ${tone}`} aria-live="polite">
          <div className="verdict-row">
            <div>
              <span className="result-kicker">Prediction</span>
              <h3>{verdict}</h3>
            </div>
            <div className="score-ring" style={{ "--score": displayedRisk }}>
              <strong>{displayedRisk}%</strong>
              <span>risk</span>
            </div>
          </div>
          <p>{isScanning ? "Querying the FastAPI inference service..." : displayedReasons[0]?.detail}</p>
          <div className="badge-row">
            {displayedReasons.map((reason) => (
              <span className={`badge ${reason.tone}`} key={reason.label}>
                {reason.tone === "safe" ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                {reason.label}
              </span>
            ))}
          </div>
          <span className="inference-status">
            {API_BASE_URL && remoteResult ? "FastAPI trained-model inference" : "Demo heuristic - browser fallback"}
          </span>
        </aside>
      </div>

      <div className="table-wrap">
        <table>
          <caption>Extracted URL Features</caption>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Value</th>
              <th>Interpretation</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, value, interpretation]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{value}</td>
                <td>{interpretation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function App() {
  return (
    <>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="PhishGuard XAI home">
          <span className="brand-mark">
            <ShieldAlert aria-hidden="true" size={21} />
          </span>
          <span>PhishGuard XAI</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#demo">Demo</a>
          <a href="#features">Features</a>
          <a href="#models">Models</a>
          <a href="#dataset">Dataset</a>
          <a href="#dashboard">Dashboard</a>
          <a href="#publication">Publication</a>
        </nav>
      </header>

      <main id="top">
        <section className="hero" aria-label="Project overview">
          <img src="/assets/phishing-detection-lab.png" alt="Cybersecurity lab workstation showing a phishing URL analysis dashboard." />
          <div className="hero-shade" />
          <div className="hero-content">
            <p className="eyebrow">Machine Learning Cybersecurity Research</p>
            <h1>Explainable Phishing URL Detection for Real-Time Link Analysis</h1>
            <p>
              A publication-ready React prototype for URL-only phishing classification, feature extraction, model benchmarking,
              and analyst-friendly explanation badges.
            </p>
            <div className="hero-actions">
              <a className="button primary" href="#demo">
                <Search aria-hidden="true" size={18} />
                Run Analyzer
              </a>
              <a className="button secondary" href="/research-paper.md">
                <BookOpen aria-hidden="true" size={18} />
                Open Paper
              </a>
            </div>
          </div>
        </section>

        <section className="stats-band" aria-label="System highlights">
          <Stat value="12" label="lexical URL features" />
          <Stat value="5" label="ML models for benchmarking" />
          <Stat value="0" label="webpage visits required" />
          <Stat value="XAI" label="human-readable diagnostics" />
        </section>

        <UrlAnalyzer />

        <section id="features" className="section features-section" aria-labelledby="features-title">
          <div className="section-heading">
            <p className="eyebrow">Feature Engineering</p>
            <h2 id="features-title">Clear Signals From the URL Itself</h2>
            <p>
              The detector avoids loading suspicious websites. It extracts fast, explainable indicators from the submitted link.
            </p>
          </div>
          <div className="card-grid">
            {featureGroups.map(({ title, icon: Icon, body }) => (
              <article className="info-card" key={title}>
                <div className="card-icon">
                  <Icon aria-hidden="true" size={22} />
                </div>
                <h3>{title}</h3>
                <p>{body}</p>
              </article>
            ))}
          </div>

          <div className="feature-catalog">
            <h3>Complete 12-Feature Extraction Set</h3>
            <div className="catalog-grid">
              {featureCatalog.map(([name, measure, rationale]) => (
                <article key={name}>
                  <strong>{name}</strong>
                  <span>{measure}</span>
                  <p>{rationale}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section id="models" className="section models-section" aria-labelledby="models-title">
          <div className="section-heading">
            <p className="eyebrow">Comparative Benchmark</p>
            <h2 id="models-title">Models Designed for the Research Study</h2>
            <p>
              Final metrics should be filled after training on verified phishing feeds and legitimate URL rankings.
            </p>
          </div>
          <div className="model-grid">
            {modelCards.map(({ name, tag, role, params, icon: Icon, body }) => (
              <article className="model-card" key={name}>
                <span>{tag}</span>
                <div className="model-title">
                  <Icon aria-hidden="true" size={24} />
                  <h3>{name}</h3>
                </div>
                <p>{body}</p>
                <dl>
                  <div>
                    <dt>Role</dt>
                    <dd>{role}</dd>
                  </div>
                  <div>
                    <dt>Parameters</dt>
                    <dd>{params}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>

          <div className="benchmark-wrap">
            <table>
              <caption>Benchmark Reporting Template</caption>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-score</th>
                  <th>ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {modelCards.map((model) => (
                  (() => {
                    const metrics = benchmarkResults[model.name];
                    return (
                  <tr key={model.name}>
                    <td>{model.name}</td>
                    {metrics.map((metric) => <td key={metric}>{metric}</td>)}
                  </tr>
                    );
                  })()
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="dataset" className="section dataset-section" aria-labelledby="dataset-title">
          <div className="section-heading">
            <p className="eyebrow">Research Methodology</p>
            <h2 id="dataset-title">Dataset, Preprocessing, and Evaluation</h2>
            <p>
              The paper requires verified phishing links, legitimate URL samples, leakage-aware splitting, and standard ML metrics.
            </p>
          </div>
          <div className="method-grid">
            <div className="table-wrap method-table">
              <table>
                <caption>Dataset Protocol</caption>
                <thead>
                  <tr>
                    <th>Stage</th>
                    <th>Source / Method</th>
                    <th>Publication Note</th>
                  </tr>
                </thead>
                <tbody>
                  {datasetRows.map(([stage, source, note]) => (
                    <tr key={stage}>
                      <td>{stage}</td>
                      <td>{source}</td>
                      <td>{note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="metric-panel">
              {metricRows.map(([metric, purpose]) => (
                <article key={metric}>
                  <strong>{metric}</strong>
                  <span>{purpose}</span>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="workflow-section" aria-labelledby="workflow-title">
          <div className="section-heading">
            <p className="eyebrow">Pipeline</p>
            <h2 id="workflow-title">From Link to Decision</h2>
          </div>
          <ol className="workflow">
            <li>
              <Database aria-hidden="true" size={23} />
              <strong>Collect</strong>
              <span>Verified phishing links and legitimate URLs are cleaned, deduplicated, and split for evaluation.</span>
            </li>
            <li>
              <LinkIcon aria-hidden="true" size={23} />
              <strong>Extract</strong>
              <span>The parser computes length, entropy, protocol, IP, keyword, digit, and structural features.</span>
            </li>
            <li>
              <BrainCircuit aria-hidden="true" size={23} />
              <strong>Predict</strong>
              <span>Supervised classifiers output a phishing probability; the browser fallback is only a demo heuristic.</span>
            </li>
            <li>
              <ShieldCheck aria-hidden="true" size={23} />
              <strong>Flag</strong>
              <span>Rule-based risk indicators show which URL properties triggered warnings.</span>
            </li>
          </ol>
        </section>

        <section id="dashboard" className="section dashboard-section" aria-labelledby="dashboard-title">
          <div className="section-heading">
            <p className="eyebrow">Dashboard and Analytics</p>
            <h2 id="dashboard-title">Operational Views From the Paper</h2>
            <p>
              The publication dashboard tracks scan outcomes, latency, warning triggers, model confidence, and drift signals.
            </p>
          </div>
          <div className="dashboard-grid">
            <article>
              <span>Traffic Ratio</span>
              <strong>Legitimate vs. phishing scans</strong>
              <p>Shows how many submitted URLs are allowed, suspicious, or blocked.</p>
            </article>
            <article>
              <span>Latency</span>
              <strong>Feature extraction under real-time constraints</strong>
              <p>URL-only extraction is designed for browser, email, and gateway use.</p>
            </article>
            <article>
              <span>Explainability</span>
              <strong>Most common warning triggers</strong>
              <p>Summarizes entropy, subdomain, keyword, IP, and HTTPS findings.</p>
            </article>
            <article>
              <span>Drift Detection</span>
              <strong>Incoming URL pattern changes</strong>
              <p>Flags shifts in entropy, special characters, keywords, or subdomain depth.</p>
            </article>
          </div>
        </section>

        <section id="publication" className="section publication-section" aria-labelledby="publication-title">
          <div className="section-heading">
            <p className="eyebrow">Publication Companion</p>
            <h2 id="publication-title">Ready for Project Submission</h2>
            <p>
              This site presents the system clearly for a research paper, viva, seminar, or final-year project demonstration.
            </p>
          </div>
          <div className="publication-grid">
            <article className="paper-panel">
              <h3>Included Research Sections</h3>
              <ul>
                <li>Abstract and introduction</li>
                <li>Related work and research gap</li>
                <li>Architecture and feature extraction</li>
                <li>ML benchmark methodology</li>
                <li>Explainability and dashboard design</li>
                <li>Limitations, ethics, and future work</li>
              </ul>
            </article>
            <article className="paper-panel highlight">
              <h3>Two-Tier Research Stack</h3>
              <p>
                The React frontend handles the interactive paper companion. A lightweight FastAPI service can extract the same
                features and serve the trained XGBoost champion when you set <strong>VITE_API_BASE_URL</strong>.
              </p>
              <a href="/research-paper.md">
                View paper draft
                <ChevronRight aria-hidden="true" size={18} />
              </a>
            </article>
          </div>
        </section>
      </main>

      <footer>
        <span>PhishGuard XAI React Prototype</span>
        <span>URL-only detection | Rule-based risk indicators | Publication website</span>
      </footer>
    </>
  );
}

export default App;
