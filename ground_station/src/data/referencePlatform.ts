export type EvidenceStatus = "complete" | "draft" | "scaffold" | "missing";

export interface EvidenceItem {
  id: string;
  standard: string;
  title: string;
  status: EvidenceStatus;
  artifacts: number;
}

export const aircraft = {
  name: "Canadian Reference VTOL UAS",
  className: "Small VTOL <25 kg",
  mode: "Safety-filtered dry run",
  airspeedMps: 21.8,
  altitudeM: 85,
  enduranceMin: 90,
  batteryPercent: 76,
  linkMarginDb: 17.5,
  containment: "Contained BVLOS test volume",
  approvalClaim: "Evidence readiness only",
};

export const route = [
  { label: "Launch", x: 14, y: 74 },
  { label: "C2 Check", x: 30, y: 58 },
  { label: "Payload Sweep", x: 51, y: 45 },
  { label: "Hold", x: 69, y: 34 },
  { label: "Recovery", x: 83, y: 52 },
];

export const payloads = [
  {
    name: "EO/IR gimbal",
    status: "Nominal",
    massKg: 0.95,
    powerW: 24,
    products: "RGB, thermal, health",
  },
  {
    name: "Depth/range module",
    status: "Nominal",
    massKg: 0.45,
    powerW: 14,
    products: "Range points, obstacles",
  },
  {
    name: "Configurable payload slot",
    status: "Review gate",
    massKg: 0.8,
    powerW: 25,
    products: "Protected B ceiling",
  },
];

export const autonomyDecision = {
  recommendation: "Proceed to payload sweep leg",
  selectedAction: [3.0, 0.0, 0.0, 0.0],
  safety: "Accepted after supervisor filter",
  surprise: 0.13,
  candidateCount: 128,
  acceptedCandidates: 118,
  operatorAuthority: true,
};

export const pipeline = {
  bundleId: "crdp-demo-bundle-001",
  exportStatus: "Review required",
  manifestHash: "62df5a8b...b18c",
  encryptionStatus: "External KMS required",
  blockedLabels: ["classified", "controlled_goods", "secret", "top_secret"],
  gates: [
    "hash chain",
    "model provenance",
    "export control",
    "audit review",
    "classified exclusion",
  ],
};

export const evidenceItems: EvidenceItem[] = [
  {
    id: "922-08",
    standard: "Standard 922.08",
    title: "Containment and operational volume",
    status: "draft",
    artifacts: 3,
  },
  {
    id: "922-09",
    standard: "Standard 922.09",
    title: "C2 link and lost-link behavior",
    status: "draft",
    artifacts: 3,
  },
  {
    id: "922-10",
    standard: "Standard 922.10",
    title: "DAA assumptions",
    status: "scaffold",
    artifacts: 2,
  },
  {
    id: "922-11",
    standard: "Standard 922.11",
    title: "Control station operator awareness",
    status: "draft",
    artifacts: 3,
  },
  {
    id: "922-12",
    standard: "Standard 922.12",
    title: "Environmental envelope",
    status: "missing",
    artifacts: 2,
  },
  {
    id: "SEC",
    standard: "Program security",
    title: "Classified path and export gates",
    status: "draft",
    artifacts: 3,
  },
  {
    id: "NK",
    standard: "Safety posture",
    title: "Non-kinetic autonomy assurance",
    status: "complete",
    artifacts: 2,
  },
];

export const logEvents = [
  { time: "14:02:11", label: "Observation received", state: "verified" },
  { time: "14:02:12", label: "128 candidates generated", state: "verified" },
  { time: "14:02:12", label: "Safety supervisor accepted 118", state: "verified" },
  { time: "14:02:13", label: "Operator approved dry-run setpoint", state: "verified" },
  { time: "14:02:13", label: "Bundle manifest requires review", state: "queued" },
];
