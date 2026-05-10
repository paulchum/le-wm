import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  Archive,
  BadgeCheck,
  Gauge,
  Layers,
  LockKeyhole,
  Map,
  Plane,
  Radar,
  ShieldCheck,
  SlidersHorizontal,
  TerminalSquare,
} from "lucide-react";
import {
  aircraft,
  autonomyDecision,
  evidenceItems,
  logEvents,
  payloads,
  pipeline,
  route,
  type EvidenceItem,
  type EvidenceStatus,
} from "./data/referencePlatform";

const navItems = [
  { icon: Map, label: "Mission" },
  { icon: Plane, label: "Aircraft" },
  { icon: Radar, label: "Payloads" },
  { icon: LockKeyhole, label: "Pipeline" },
  { icon: BadgeCheck, label: "Evidence" },
  { icon: TerminalSquare, label: "Logs" },
];

const statusTone: Record<EvidenceStatus, string> = {
  complete: "good",
  draft: "working",
  scaffold: "queued",
  missing: "risk",
};

export default function App() {
  const [selectedEvidence, setSelectedEvidence] = useState(evidenceItems[0]);
  const [mapMode, setMapMode] = useState<"route" | "containment">("route");
  const readiness = useMemo(() => {
    const completed = evidenceItems.filter((item) => item.status === "complete").length;
    return Math.round((completed / evidenceItems.length) * 100);
  }, []);

  return (
    <div className="app-shell">
      <aside className="nav-rail" aria-label="Ground station navigation">
        <div className="brand-mark" aria-label="CRDP">
          C
        </div>
        <div className="nav-icons">
          {navItems.map(({ icon: Icon, label }) => (
            <button key={label} className="nav-button" title={label} aria-label={label}>
              <Icon size={19} strokeWidth={1.9} />
            </button>
          ))}
        </div>
      </aside>

      <main className="console">
        <header className="top-bar">
          <div>
            <h1>Canadian Reference Drone Platform</h1>
            <p>{aircraft.className} · BVLOS/L1C evidence readiness · non-kinetic</p>
          </div>
          <div className="top-status">
            <Metric label="Mode" value={aircraft.mode} tone="working" />
            <Metric label="Link" value={`${aircraft.linkMarginDb} dB`} tone="good" />
            <Metric label="Claim" value={aircraft.approvalClaim} tone="queued" />
          </div>
        </header>

        <section className="status-strip" aria-label="Aircraft status">
          <Metric label="Airspeed" value={`${aircraft.airspeedMps} m/s`} icon={<Gauge />} />
          <Metric label="Altitude" value={`${aircraft.altitudeM} m`} icon={<Activity />} />
          <Metric label="Battery" value={`${aircraft.batteryPercent}%`} tone="good" />
          <Metric label="Endurance" value={`${aircraft.enduranceMin} min`} />
          <Metric label="Evidence" value={`${readiness}% complete`} tone="working" />
        </section>

        <div className="work-grid">
          <section className="panel map-panel">
            <PanelHeader
              icon={<Map />}
              title="Contained BVLOS Mission"
              action={
                <div className="segmented">
                  <button
                    className={mapMode === "route" ? "selected" : ""}
                    onClick={() => setMapMode("route")}
                  >
                    Route
                  </button>
                  <button
                    className={mapMode === "containment" ? "selected" : ""}
                    onClick={() => setMapMode("containment")}
                  >
                    Volume
                  </button>
                </div>
              }
            />
            <MissionMap mode={mapMode} />
          </section>

          <section className="panel autonomy-panel">
            <PanelHeader icon={<ShieldCheck />} title="Autonomy Recommendation" />
            <div className="recommendation">
              <strong>{autonomyDecision.recommendation}</strong>
              <span>{autonomyDecision.safety}</span>
            </div>
            <div className="decision-grid">
              <Metric
                label="Candidates"
                value={`${autonomyDecision.acceptedCandidates}/${autonomyDecision.candidateCount}`}
              />
              <Metric label="Surprise" value={autonomyDecision.surprise.toFixed(2)} tone="good" />
              <Metric label="Action" value={autonomyDecision.selectedAction.join(", ")} />
            </div>
            <label className="toggle-row">
              <input type="checkbox" checked={autonomyDecision.operatorAuthority} readOnly />
              Operator authority held
            </label>
          </section>

          <section className="panel payload-panel">
            <PanelHeader icon={<Radar />} title="Payload Suite" />
            <div className="payload-list">
              {payloads.map((payload) => (
                <div className="payload-row" key={payload.name}>
                  <div>
                    <strong>{payload.name}</strong>
                    <span>{payload.products}</span>
                  </div>
                  <div className="payload-meta">
                    <span>{payload.massKg} kg</span>
                    <span>{payload.powerW} W</span>
                    <b>{payload.status}</b>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="panel pipeline-panel">
            <PanelHeader icon={<LockKeyhole />} title="Secure Data Pipeline" />
            <div className="pipeline-id">{pipeline.bundleId}</div>
            <div className="pipeline-grid">
              <Metric label="Export" value={pipeline.exportStatus} tone="working" />
              <Metric label="Encryption" value={pipeline.encryptionStatus} tone="queued" />
              <Metric label="Manifest" value={pipeline.manifestHash} />
            </div>
            <div className="gate-list">
              {pipeline.gates.map((gate) => (
                <span key={gate}>{gate}</span>
              ))}
            </div>
          </section>

          <section className="panel evidence-panel">
            <PanelHeader icon={<BadgeCheck />} title="Standard 922 Evidence Matrix" />
            <div className="evidence-grid">
              <div className="evidence-list" role="list">
                {evidenceItems.map((item) => (
                  <button
                    key={item.id}
                    className={item.id === selectedEvidence.id ? "evidence-row selected" : "evidence-row"}
                    onClick={() => setSelectedEvidence(item)}
                  >
                    <span>{item.id}</span>
                    <strong>{item.title}</strong>
                    <em className={`tone-${statusTone[item.status]}`}>{item.status}</em>
                  </button>
                ))}
              </div>
              <EvidenceDetail item={selectedEvidence} />
            </div>
          </section>

          <section className="panel log-panel">
            <PanelHeader icon={<Archive />} title="Mission Log Verification" />
            <div className="log-list">
              {logEvents.map((event) => (
                <div className="log-row" key={`${event.time}-${event.label}`}>
                  <span>{event.time}</span>
                  <strong>{event.label}</strong>
                  <em className={event.state}>{event.state}</em>
                </div>
              ))}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

function PanelHeader({
  icon,
  title,
  action,
}: {
  icon: ReactNode;
  title: string;
  action?: ReactNode;
}) {
  return (
    <div className="panel-header">
      <div>
        <span className="header-icon">{icon}</span>
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  );
}

function Metric({
  label,
  value,
  tone,
  icon,
}: {
  label: string;
  value: string;
  tone?: "good" | "working" | "queued";
  icon?: ReactNode;
}) {
  return (
    <div className={`metric ${tone ? `metric-${tone}` : ""}`}>
      {icon ? <span className="metric-icon">{icon}</span> : null}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MissionMap({ mode }: { mode: "route" | "containment" }) {
  return (
    <div className={`mission-map ${mode}`}>
      <div className="map-grid-lines" />
      <div className="containment-ring outer" />
      <div className="containment-ring inner" />
      <svg className="route-line" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline
          points={route.map((point) => `${point.x},${point.y}`).join(" ")}
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.8"
        />
      </svg>
      {route.map((point, index) => (
        <div
          key={point.label}
          className={`waypoint ${index === 2 ? "active" : ""}`}
          style={{ left: `${point.x}%`, top: `${point.y}%` }}
        >
          <span>{index + 1}</span>
          <strong>{point.label}</strong>
        </div>
      ))}
      <div className="aircraft-marker" style={{ left: "45%", top: "48%" }}>
        <Plane size={22} />
      </div>
      <div className="map-caption">
        <Layers size={16} />
        {mode === "route" ? aircraft.containment : "Emergency containment volume displayed"}
      </div>
    </div>
  );
}

function EvidenceDetail({ item }: { item: EvidenceItem }) {
  const artifactLabel = item.artifacts === 1 ? "artifact" : "artifacts";

  return (
    <div className="evidence-detail">
      <div className={`status-dot tone-${statusTone[item.status]}`} />
      <span>{item.standard}</span>
      <h3>{item.title}</h3>
      <p>
        {item.artifacts} {artifactLabel} tracked for this evidence row. Release remains blocked until
        missing and classified-path gates are cleared.
      </p>
      <button className="detail-action">
        <SlidersHorizontal size={16} />
        Review artifacts
      </button>
    </div>
  );
}
