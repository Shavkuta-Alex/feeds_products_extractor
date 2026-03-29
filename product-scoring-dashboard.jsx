import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from "recharts";

const data = {
  training: { auc: 0.9976, precision: 0.968, recall: 0.962, f1: 0.965, accuracy: 0.976 },
  labeled: {
    total: 5000, predicted: 1723, actual: 1723,
    categories: [
      { name: "Pet Supplies", total: 502, selling: 196, avg: 45.7 },
      { name: "Sports", total: 492, selling: 177, avg: 42.5 },
      { name: "Beauty", total: 493, selling: 177, avg: 42.2 },
      { name: "Electronics", total: 555, selling: 190, avg: 41.4 },
      { name: "Home", total: 497, selling: 171, avg: 41.2 },
      { name: "Automotive", total: 476, selling: 163, avg: 41.2 },
      { name: "Toys", total: 469, selling: 155, avg: 40.7 },
      { name: "Clothing", total: 496, selling: 166, avg: 40.4 },
      { name: "Books", total: 523, selling: 171, avg: 39.6 },
      { name: "Garden", total: 497, selling: 157, avg: 39.2 },
    ],
  },
  unlabeled: {
    total: 3000, predicted: 876, actual: 884, accuracy: 0.9813,
    categories: [
      { name: "Pet Supplies", total: 266, selling: 82, avg: 39.5 },
      { name: "Garden", total: 335, selling: 104, avg: 38.6 },
      { name: "Automotive", total: 336, selling: 100, avg: 38.2 },
      { name: "Sports", total: 309, selling: 93, avg: 37.9 },
      { name: "Electronics", total: 311, selling: 91, avg: 37.8 },
      { name: "Clothing", total: 282, selling: 83, avg: 37.4 },
      { name: "Home", total: 312, selling: 94, avg: 37.2 },
      { name: "Beauty", total: 271, selling: 75, avg: 36.4 },
      { name: "Books", total: 298, selling: 81, avg: 35.4 },
      { name: "Toys", total: 280, selling: 73, avg: 35.0 },
    ],
  },
  features: [
    { name: "Description Length", imp: 0.643 },
    { name: "Price Z-Score", imp: 0.137 },
    { name: "Bullet Points", imp: 0.125 },
    { name: "Image Count", imp: 0.029 },
    { name: "Days Since Modified", imp: 0.028 },
    { name: "Desc Word Count", imp: 0.014 },
    { name: "Title Caps Ratio", imp: 0.005 },
    { name: "Price (log)", imp: 0.003 },
    { name: "Title Length", imp: 0.003 },
  ],
  topProducts: [
    { id: "PRD-275163", title: "Purina Max Pet Product", cat: "Pet Supplies", price: 38.99, score: 98.3, imgs: 8, variants: 4 },
    { id: "PRD-485111", title: "Yeti Eco Sports Product", cat: "Sports", price: 68.99, score: 98.2, imgs: 8, variants: 2 },
    { id: "PRD-965542", title: "Armor All Classic Auto", cat: "Automotive", price: 41.99, score: 98.2, imgs: 8, variants: 4 },
    { id: "PRD-869769", title: "Mattel Eco Toys Product", cat: "Toys", price: 39.99, score: 98.1, imgs: 8, variants: 2 },
    { id: "PRD-758962", title: "Fisher-Price Pro Toys", cat: "Toys", price: 33.98, score: 98.1, imgs: 8, variants: 2 },
  ],
  bottomProducts: [
    { id: "PRD-370069", title: "Eco Garden Product", cat: "Garden", price: 15.94, score: 6.3, imgs: 0, variants: 1 },
    { id: "PRD-680053", title: "Classic Sports Product", cat: "Sports", price: 22.99, score: 6.8, imgs: 0, variants: 1 },
    { id: "PRD-953956", title: "Max Beauty Product", cat: "Beauty", price: 72.14, score: 6.9, imgs: 1, variants: 2 },
    { id: "PRD-202319", title: "Pro Garden Product", cat: "Garden", price: 22.94, score: 6.9, imgs: 1, variants: 2 },
    { id: "PRD-445705", title: "Eco Clothing Product", cat: "Clothing", price: 94.07, score: 7.3, imgs: 0, variants: 3 },
  ],
};

const COLORS = {
  bg: "#0a0e17",
  card: "#111827",
  cardBorder: "#1e293b",
  accent: "#22d3ee",
  accentDim: "#0e7490",
  green: "#34d399",
  greenDim: "#065f46",
  red: "#f87171",
  redDim: "#7f1d1d",
  amber: "#fbbf24",
  text: "#e2e8f0",
  textDim: "#94a3b8",
  textMuted: "#64748b",
};

const Card = ({ children, style }) => (
  <div style={{
    background: COLORS.card, border: `1px solid ${COLORS.cardBorder}`,
    borderRadius: 12, padding: "20px 24px", ...style,
  }}>{children}</div>
);

const Metric = ({ label, value, sub, color }) => (
  <div style={{ textAlign: "center" }}>
    <div style={{ fontSize: 11, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 6 }}>{label}</div>
    <div style={{ fontSize: 32, fontWeight: 700, color: color || COLORS.accent, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
    {sub && <div style={{ fontSize: 12, color: COLORS.textDim, marginTop: 4 }}>{sub}</div>}
  </div>
);

const Badge = ({ children, color = COLORS.accent }) => (
  <span style={{
    display: "inline-block", padding: "2px 10px", borderRadius: 100,
    fontSize: 11, fontWeight: 600, background: color + "20", color,
    border: `1px solid ${color}40`,
  }}>{children}</span>
);

const ScoreBar = ({ score }) => {
  const color = score >= 80 ? COLORS.green : score >= 50 ? COLORS.amber : COLORS.red;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ width: 80, height: 6, background: COLORS.cardBorder, borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${score}%`, height: "100%", background: color, borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: 13, fontWeight: 600, color, fontFamily: "'JetBrains Mono', monospace", minWidth: 36 }}>{score}</span>
    </div>
  );
};

const tabs = ["Overview", "Model Performance", "Categories", "Products", "How It Works"];

export default function Dashboard() {
  const [tab, setTab] = useState(0);

  return (
    <div style={{
      minHeight: "100vh", background: COLORS.bg, color: COLORS.text,
      fontFamily: "'Inter', -apple-system, sans-serif",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ padding: "28px 32px 0", borderBottom: `1px solid ${COLORS.cardBorder}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <div style={{ width: 10, height: 10, borderRadius: "50%", background: COLORS.accent, boxShadow: `0 0 12px ${COLORS.accent}80` }} />
          <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0, letterSpacing: -0.5 }}>Product Selling Scorer</h1>
          <Badge color={COLORS.green}>Pipeline Active</Badge>
        </div>
        <p style={{ fontSize: 13, color: COLORS.textDim, margin: "4px 0 16px" }}>
          Predict selling products from basic feed data — no ratings or sales history needed
        </p>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 0 }}>
          {tabs.map((t, i) => (
            <button key={t} onClick={() => setTab(i)} style={{
              padding: "10px 20px", fontSize: 13, fontWeight: tab === i ? 600 : 400,
              color: tab === i ? COLORS.accent : COLORS.textDim,
              background: "none", border: "none", cursor: "pointer",
              borderBottom: tab === i ? `2px solid ${COLORS.accent}` : "2px solid transparent",
              transition: "all 0.15s",
            }}>{t}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: "24px 32px", maxWidth: 1100 }}>
        {tab === 0 && <OverviewTab />}
        {tab === 1 && <ModelTab />}
        {tab === 2 && <CategoriesTab />}
        {tab === 3 && <ProductsTab />}
        {tab === 4 && <HowItWorksTab />}
      </div>
    </div>
  );
}

function OverviewTab() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        <Card><Metric label="ML Accuracy" value="98.1%" sub="on unlabeled feed" color={COLORS.green} /></Card>
        <Card><Metric label="AUC-ROC" value="0.997" sub="model quality" color={COLORS.accent} /></Card>
        <Card><Metric label="Products Scored" value="8,000" sub="across 2 feeds" /></Card>
        <Card><Metric label="Selling Found" value="2,599" sub="~32% of total" color={COLORS.amber} /></Card>
      </div>

      {/* Two-column layout */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Card>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: COLORS.textDim }}>Feature Importance</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.features} layout="vertical" margin={{ left: 10, right: 20 }}>
              <XAxis type="number" tick={{ fill: COLORS.textMuted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" tick={{ fill: COLORS.textDim, fontSize: 11 }} width={120} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.cardBorder}`, borderRadius: 8, color: COLORS.text, fontSize: 12 }} />
              <Bar dataKey="imp" radius={[0, 4, 4, 0]}>
                {data.features.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? COLORS.accent : i < 3 ? COLORS.accentDim : COLORS.cardBorder} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: COLORS.textDim }}>Pipeline Architecture</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {[
              { step: "1", label: "Ingest Feed", desc: "Parse product data (title, price, images, etc.)" },
              { step: "2", label: "Extract Features", desc: "30+ signals from basic fields" },
              { step: "3", label: "Heuristic Score", desc: "Rule-based 0-100 score (always available)" },
              { step: "4", label: "ML Classify", desc: "Trained on labeled feeds, applied to unlabeled" },
              { step: "5", label: "Combined Score", desc: "70% ML + 30% heuristic → final prediction" },
            ].map(s => (
              <div key={s.step} style={{ display: "flex", gap: 14, alignItems: "center" }}>
                <div style={{
                  width: 32, height: 32, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center",
                  background: `${COLORS.accent}15`, color: COLORS.accent, fontSize: 13, fontWeight: 700, fontFamily: "'JetBrains Mono'",
                  border: `1px solid ${COLORS.accent}30`, flexShrink: 0,
                }}>{s.step}</div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{s.label}</div>
                  <div style={{ fontSize: 11, color: COLORS.textMuted }}>{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Feed comparison */}
      <Card>
        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: COLORS.textDim }}>Feed Comparison</h3>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>Labeled Feed (Training)</span>
              <Badge color={COLORS.green}>Has Sales Data</Badge>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
              <div style={{ padding: 12, background: `${COLORS.accent}08`, borderRadius: 8, textAlign: "center" }}>
                <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "'JetBrains Mono'" }}>5,000</div>
                <div style={{ fontSize: 10, color: COLORS.textMuted }}>TOTAL</div>
              </div>
              <div style={{ padding: 12, background: `${COLORS.green}08`, borderRadius: 8, textAlign: "center" }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: COLORS.green, fontFamily: "'JetBrains Mono'" }}>1,723</div>
                <div style={{ fontSize: 10, color: COLORS.textMuted }}>SELLING</div>
              </div>
              <div style={{ padding: 12, background: `${COLORS.red}08`, borderRadius: 8, textAlign: "center" }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: COLORS.red, fontFamily: "'JetBrains Mono'" }}>3,277</div>
                <div style={{ fontSize: 10, color: COLORS.textMuted }}>NOT SELLING</div>
              </div>
            </div>
          </div>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>Unlabeled Feed (Prediction)</span>
              <Badge color={COLORS.amber}>No Sales Data</Badge>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
              <div style={{ padding: 12, background: `${COLORS.accent}08`, borderRadius: 8, textAlign: "center" }}>
                <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "'JetBrains Mono'" }}>3,000</div>
                <div style={{ fontSize: 10, color: COLORS.textMuted }}>TOTAL</div>
              </div>
              <div style={{ padding: 12, background: `${COLORS.green}08`, borderRadius: 8, textAlign: "center" }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: COLORS.green, fontFamily: "'JetBrains Mono'" }}>876</div>
                <div style={{ fontSize: 10, color: COLORS.textMuted }}>PREDICTED</div>
              </div>
              <div style={{ padding: 12, background: `${COLORS.accent}08`, borderRadius: 8, textAlign: "center" }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: COLORS.accent, fontFamily: "'JetBrains Mono'" }}>98.1%</div>
                <div style={{ fontSize: 10, color: COLORS.textMuted }}>ACCURACY</div>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function ModelTab() {
  const metrics = [
    { label: "Precision", value: data.training.precision, desc: "Of predicted sellers, how many actually are" },
    { label: "Recall", value: data.training.recall, desc: "Of actual sellers, how many we caught" },
    { label: "F1 Score", value: data.training.f1, desc: "Harmonic mean of precision & recall" },
    { label: "Accuracy", value: data.training.accuracy, desc: "Overall correct predictions" },
    { label: "AUC-ROC", value: data.training.auc, desc: "Area under ROC curve" },
  ];

  const confMatrix = [
    { label: "True Positive", value: "~1,658", desc: "Correctly identified sellers", color: COLORS.green },
    { label: "False Positive", value: "~55", desc: "Non-sellers flagged as sellers", color: COLORS.amber },
    { label: "False Negative", value: "~66", desc: "Sellers we missed", color: COLORS.red },
    { label: "True Negative", value: "~3,221", desc: "Correctly identified non-sellers", color: COLORS.accent },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Card>
        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 20, color: COLORS.textDim }}>Classification Metrics</h3>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${metrics.length}, 1fr)`, gap: 16 }}>
          {metrics.map(m => (
            <div key={m.label} style={{ textAlign: "center", padding: 16, background: `${COLORS.accent}06`, borderRadius: 10 }}>
              <div style={{ fontSize: 28, fontWeight: 700, color: m.value >= 0.97 ? COLORS.green : COLORS.accent, fontFamily: "'JetBrains Mono'" }}>
                {m.value.toFixed(3)}
              </div>
              <div style={{ fontSize: 12, fontWeight: 600, marginTop: 6 }}>{m.label}</div>
              <div style={{ fontSize: 10, color: COLORS.textMuted, marginTop: 4 }}>{m.desc}</div>
            </div>
          ))}
        </div>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Card>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: COLORS.textDim }}>Confusion Matrix (approx.)</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {confMatrix.map(c => (
              <div key={c.label} style={{
                padding: 16, borderRadius: 10, background: `${c.color}10`,
                border: `1px solid ${c.color}25`, textAlign: "center",
              }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: c.color, fontFamily: "'JetBrains Mono'" }}>{c.value}</div>
                <div style={{ fontSize: 11, fontWeight: 600, marginTop: 4 }}>{c.label}</div>
                <div style={{ fontSize: 10, color: COLORS.textMuted, marginTop: 2 }}>{c.desc}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: COLORS.textDim }}>Top Features by Importance</h3>
          {data.features.slice(0, 7).map((f, i) => (
            <div key={f.name} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
              <span style={{ fontSize: 11, color: COLORS.textMuted, width: 16, textAlign: "right", fontFamily: "'JetBrains Mono'" }}>{i + 1}</span>
              <span style={{ fontSize: 12, flex: 1 }}>{f.name}</span>
              <div style={{ width: 140, height: 8, background: COLORS.cardBorder, borderRadius: 4, overflow: "hidden" }}>
                <div style={{
                  width: `${(f.imp / data.features[0].imp) * 100}%`, height: "100%",
                  background: i === 0 ? COLORS.accent : i < 3 ? COLORS.accentDim : COLORS.textMuted,
                  borderRadius: 4, transition: "width 0.5s",
                }} />
              </div>
              <span style={{ fontSize: 11, color: COLORS.textMuted, fontFamily: "'JetBrains Mono'", width: 40, textAlign: "right" }}>
                {(f.imp * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
}

function CategoriesTab() {
  const [feed, setFeed] = useState("labeled");
  const cats = feed === "labeled" ? data.labeled.categories : data.unlabeled.categories;
  const chartData = cats.map(c => ({
    name: c.name,
    selling: c.selling,
    notSelling: c.total - c.selling,
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", gap: 8 }}>
        {["labeled", "unlabeled"].map(f => (
          <button key={f} onClick={() => setFeed(f)} style={{
            padding: "8px 18px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer",
            background: feed === f ? COLORS.accent + "20" : "transparent",
            color: feed === f ? COLORS.accent : COLORS.textMuted,
            border: `1px solid ${feed === f ? COLORS.accent + "50" : COLORS.cardBorder}`,
          }}>{f === "labeled" ? "Labeled Feed" : "Unlabeled Feed"}</button>
        ))}
      </div>

      <Card>
        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: COLORS.textDim }}>Selling vs Not Selling by Category</h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} margin={{ left: 10, right: 20, bottom: 10 }}>
            <XAxis dataKey="name" tick={{ fill: COLORS.textDim, fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: COLORS.textMuted, fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.cardBorder}`, borderRadius: 8, color: COLORS.text, fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="selling" stackId="a" fill={COLORS.green} radius={[0, 0, 0, 0]} />
            <Bar dataKey="notSelling" stackId="a" fill={COLORS.cardBorder} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card>
        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: COLORS.textDim }}>Category Details</h3>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${COLORS.cardBorder}` }}>
                {["Category", "Total", "Predicted Selling", "Sell Rate", "Avg Score"].map(h => (
                  <th key={h} style={{ padding: "10px 12px", textAlign: "left", fontSize: 11, color: COLORS.textMuted, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cats.map(c => (
                <tr key={c.name} style={{ borderBottom: `1px solid ${COLORS.cardBorder}22` }}>
                  <td style={{ padding: "10px 12px", fontWeight: 500 }}>{c.name}</td>
                  <td style={{ padding: "10px 12px", fontFamily: "'JetBrains Mono'", fontSize: 12 }}>{c.total.toLocaleString()}</td>
                  <td style={{ padding: "10px 12px" }}>
                    <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 12, color: COLORS.green }}>{c.selling}</span>
                  </td>
                  <td style={{ padding: "10px 12px" }}>
                    <Badge color={(c.selling / c.total) > 0.35 ? COLORS.green : COLORS.amber}>
                      {((c.selling / c.total) * 100).toFixed(1)}%
                    </Badge>
                  </td>
                  <td style={{ padding: "10px 12px" }}><ScoreBar score={c.avg} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function ProductsTab() {
  const [view, setView] = useState("top");
  const products = view === "top" ? data.topProducts : data.bottomProducts;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", gap: 8 }}>
        {[["top", "Top Scored"], ["bottom", "Bottom Scored"]].map(([v, label]) => (
          <button key={v} onClick={() => setView(v)} style={{
            padding: "8px 18px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer",
            background: view === v ? (v === "top" ? COLORS.green : COLORS.red) + "20" : "transparent",
            color: view === v ? (v === "top" ? COLORS.green : COLORS.red) : COLORS.textMuted,
            border: `1px solid ${view === v ? (v === "top" ? COLORS.green : COLORS.red) + "50" : COLORS.cardBorder}`,
          }}>{label}</button>
        ))}
      </div>

      {products.map(p => (
        <Card key={p.id}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                <span style={{ fontSize: 15, fontWeight: 600 }}>{p.title}</span>
                <Badge color={p.score >= 50 ? COLORS.green : COLORS.red}>
                  {p.score >= 50 ? "SELLING" : "NOT SELLING"}
                </Badge>
              </div>
              <div style={{ display: "flex", gap: 20, fontSize: 12, color: COLORS.textDim }}>
                <span>{p.cat}</span>
                <span>${p.price}</span>
                <span>{p.imgs} images</span>
                <span>{p.variants} variant{p.variants > 1 ? "s" : ""}</span>
                <span style={{ fontFamily: "'JetBrains Mono'", color: COLORS.textMuted }}>{p.id}</span>
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{
                fontSize: 28, fontWeight: 700, fontFamily: "'JetBrains Mono'",
                color: p.score >= 80 ? COLORS.green : p.score >= 50 ? COLORS.amber : COLORS.red,
              }}>{p.score}</div>
              <div style={{ fontSize: 10, color: COLORS.textMuted }}>SCORE</div>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}

function HowItWorksTab() {
  const signals = [
    { cat: "Title Signals", items: ["Word count (5-15 ideal)", "Brand name presence", "Power words (Premium, Best Seller)", "Spam signals (Wholesale, Test Listing)", "Capitalization ratio"] },
    { cat: "Description Signals", items: ["Length and word count", "Has structured bullets", "HTML formatting (active seller)", "Completeness (>20 chars)"] },
    { cat: "Price Signals", items: ["Category z-score (within 1 std dev)", "Psychological pricing (.99, .95)", "Log-scaled price", "Round number detection"] },
    { cat: "Listing Quality", items: ["Image count (1-8 scale)", "Variant/SKU count", "UPC/EAN/GTIN presence", "Overall field completeness"] },
    { cat: "Recency Signals", items: ["Days since last modified", "Modified in last 30 days", "Modified in last 90 days"] },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Card>
        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, color: COLORS.textDim }}>The Problem</h3>
        <p style={{ fontSize: 13, lineHeight: 1.7, color: COLORS.text, margin: 0 }}>
          Your store feeds have millions of products, but many feeds lack sales signals like ratings, reviews, or sold-unit counts.
          You need to identify which products are likely selling using only basic catalog fields: title, description, price, images, category, brand, and timestamps.
        </p>
      </Card>

      <Card>
        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, color: COLORS.textDim }}>The Approach</h3>
        <p style={{ fontSize: 13, lineHeight: 1.7, color: COLORS.text, margin: "0 0 16px" }}>
          The pipeline uses a two-layer approach. First, a heuristic scorer extracts 30+ signals from basic fields and computes a rule-based score.
          Second, for feeds that DO have sales data, a Gradient Boosting classifier learns the signal patterns. It then transfers that knowledge to feeds without sales data.
          The final score blends both: 70% ML + 30% heuristics.
        </p>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {signals.map(s => (
          <Card key={s.cat}>
            <h4 style={{ fontSize: 13, fontWeight: 600, color: COLORS.accent, marginBottom: 10 }}>{s.cat}</h4>
            {s.items.map(item => (
              <div key={item} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <div style={{ width: 5, height: 5, borderRadius: "50%", background: COLORS.accentDim, flexShrink: 0 }} />
                <span style={{ fontSize: 12, color: COLORS.textDim }}>{item}</span>
              </div>
            ))}
          </Card>
        ))}

        <Card>
          <h4 style={{ fontSize: 13, fontWeight: 600, color: COLORS.accent, marginBottom: 10 }}>Integration Tips</h4>
          {[
            "Process by category for best results",
            "Retrain weekly as feeds update",
            "Adjust threshold (50) per store/vertical",
            "Cross-reference UPC/GTIN for enrichment",
            "Batch process in chunks of 100K",
          ].map(item => (
            <div key={item} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", background: COLORS.accentDim, flexShrink: 0 }} />
              <span style={{ fontSize: 12, color: COLORS.textDim }}>{item}</span>
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
}
