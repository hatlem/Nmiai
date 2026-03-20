// dashboard_data.js — All competition state in one file
// Oppdater denne filen direkte (eller via script) for å oppdatere dashboardet.
// Dashboardet leser BARE herfra — ingen localStorage.

const COMPETITION_DATA = {

  // ═══════════════════════════════════════════
  // NORGESGRUPPEN — Object Detection
  // ═══════════════════════════════════════════
  norgesgruppen: {
    bestScore: 0.6740,
    rank: 110,
    totalTeams: 167,
    dailyQuotaUsed: 1,
    dailyQuotaMax: 6,
    inFlight: 0,
    submissions: [
      { id:1, date:'2026-03-19', time:'21:00', zip:'submission.zip', desc:'YOLOv8x multi-class, no classifier', score:0.4759, runtime:165.2, status:'completed', notes:'det~0.68, cls~0' },
      { id:2, date:'2026-03-19', time:'22:00', zip:'submission_v2.zip', desc:'YOLOv8x + WBF multi-scale', score:null, runtime:null, status:'failed', notes:'Exit code 1' },
      { id:3, date:'2026-03-20', time:'10:33', zip:'submission_v3.zip', desc:'YOLOv8x single-class + EfficientNet-B3', score:null, runtime:null, status:'skipped', notes:'Ikke submittet' },
      { id:4, date:'2026-03-20', time:'10:38', zip:'submission_20260320_103349.zip', desc:'YOLO26-x ONNX + DINOv2, score=det*cls^0.15', score:0.6740, runtime:287.9, status:'completed', notes:'+41.6% vs forrige' },
      { id:5, date:'2026-03-20', time:'11:52', zip:'submission_20260320_115200.zip', desc:'YOLO26-x ONNX + DINOv2 + multi-class YOLO hybrid', score:null, runtime:null, status:'failed', notes:'Exit code 1' },
      { id:6, date:'2026-03-20', time:'14:16', zip:'submission_20260320_141629.zip', desc:'YOLO26-x FP16 ONNX + DINOv2 only, pure det ranking', score:null, runtime:null, status:'ready', notes:'257MB, klar for submit' },
    ],
    leaderboard: [
      { name:'prompt injection 1678', score:0.9199, subs:8 },
      { name:'sf', score:0.9193, subs:5 },
      { name:'Benjamin Klepp', score:0.9163, subs:4 },
    ],
  },

  // ═══════════════════════════════════════════
  // TRIPLETEX — AI Accounting Agent
  // ═══════════════════════════════════════════
  tripletex: {
    bestScore: null,
    rank: null,
    totalTeams: null,
    agentUrl: 'https://tripletex-agent-174612781810.europe-north1.run.app',
    leaderboard: [
      { name:'SiddisAI', score:39.56, details:'18/30 T81 13.5' },
      { name:'Kult Byrå', score:39.31, details:'18/30 T98 13.9' },
      { name:'websecured.io', score:39.26, details:'18/30 T98 14.7' },
    ],
  },

  // ═══════════════════════════════════════════
  // ASTAR ISLAND — Viking Prediction
  // ═══════════════════════════════════════════
  astar: {
    rounds: [
      { round:1, score:55.4, rank:23, teams:117, weight:1.05, queries:50, seeds:5, status:'completed', date:'2026-03-19' },
      { round:2, score:76.9, rank:35, teams:153, weight:1.1025, queries:50, seeds:5, status:'completed', date:'2026-03-19' },
      { round:3, score:null, rank:null, teams:null, weight:1.1576, queries:0, seeds:0, status:'completed', date:'2026-03-20', note:'No submissions' },
      { round:4, score:78.8, rank:35, teams:86, weight:1.2155, queries:50, seeds:5, status:'completed', date:'2026-03-20' },
      { round:5, score:67.6, rank:68, teams:144, weight:1.2763, queries:50, seeds:5, status:'completed', date:'2026-03-20' },
      { round:6, score:60.6, rank:83, teams:186, weight:1.3401, queries:50, seeds:5, status:'completed', date:'2026-03-20' },
      { round:7, score:null, rank:null, teams:null, weight:1.4071, queries:50, seeds:5, status:'active', date:'2026-03-20', note:'50/50 queries, 5/5 submitted, venter på score' },
    ],
    leaderboard: [
      { name:'Meme Dream Team', score:118.6, details:'4 runder, 77.4' },
      { name:'Synthetic Synapses', score:117.4, details:'5 runder, 75.7' },
      { name:'Propulsion Optimizers', score:116.9, details:'6 runder, 87.7' },
    ],
  },

  // ═══════════════════════════════════════════
  // OVERALL LEADERBOARD
  // ═══════════════════════════════════════════
  overall: {
    leaderboard: [
      { name:'Kult Byrå', score:98.5, details:'TX 99.0 / AS 99.4 / NG 97.1' },
      { name:'Ignore all previous...', score:92.2, details:'TX 96.6 / AS 89.0 / NG 90.9' },
      { name:'prompt injection 1678', score:92.0, details:'TX 100.0 / AS 92.5 / NG 83.6' },
    ],
    updated: '2026-03-20T15:00:00+01:00',
  },

  // ═══════════════════════════════════════════
  // GPU TRAINING STATUS
  // ═══════════════════════════════════════════
  training: {
    vms: [
      { name:'yolo26-t4-1', gpu:'T4', zone:'eu-west1-b', task:'DINOv2 v2 (Focal+Mixup+EMA)', detail:'Ep13/50 val=90.7% mean_cat=83.4%', online:true },
      { name:'yolo26-train', gpu:'T4', zone:'eu-west1-c', task:'YOLOv8x Multi-class', detail:'Epoch 12/150, mAP50=0.786', online:true },
      { name:'yolo26-a100', gpu:'A100', zone:'us-central1-b', task:'YOLOv8x Single-class', detail:'Restarted b=2, peak 0.934', online:true },
    ],
    history: [
      { time:'2026-03-20T08:30:00Z', best:0.649, note:'VM#2 epoch 1' },
      { time:'2026-03-20T09:00:00Z', best:0.735, note:'VM#2 epoch 2' },
      { time:'2026-03-20T09:30:00Z', best:0.744, note:'VM#2 epoch 3' },
      { time:'2026-03-20T10:00:00Z', best:0.761, note:'VM#2 epoch 5' },
      { time:'2026-03-20T10:30:00Z', best:0.772, note:'VM#2 epoch 8' },
      { time:'2026-03-20T11:00:00Z', best:0.783, note:'VM#2 epoch 10' },
      { time:'2026-03-20T11:30:00Z', best:0.786, note:'VM#2 epoch 11' },
    ],
    bestMap50: 0.786,
    target: 0.920,
  },

  // ═══════════════════════════════════════════
  // HISTORIKK — Hva vi har gjort og hvorfor
  // ═══════════════════════════════════════════
  history: [
    { task:'norgesgruppen', what:'YOLOv8x multi-class detection', why:'Raskeste baseline for deteksjon', result:'det mAP~0.68, cls mAP~0. Score: 0.4759', status:'done', time:'2026-03-19T21:00:00+01:00' },
    { task:'norgesgruppen', what:'Lagt til WBF multi-scale ensemble', why:'Booste deteksjon med test-time augmentation', result:'Failed — exit code 1', status:'abandoned', time:'2026-03-19T22:00:00+01:00' },
    { task:'norgesgruppen', what:'Byttet til YOLO26-x ONNX + DINOv2 classifier', why:'YOLO26 er state-of-the-art, DINOv2 gir robust embeddings', result:'Score: 0.6740 (+41.6%)', status:'done', time:'2026-03-20T10:38:00+01:00' },
    { task:'norgesgruppen', what:'Hybrid: YOLO26 ONNX + multi-class YOLO + DINOv2', why:'Kombinere single-class + multi-class for bedre cls', result:'Failed — exit code 1', status:'abandoned', time:'2026-03-20T11:52:00+01:00' },
    { task:'norgesgruppen', what:'YOLO26 FP16 + DINOv2 only, pure det ranking', why:'Fjerne alt som kan feile, fokus på ren deteksjon', result:'Klar for submit, 257MB', status:'active', time:'2026-03-20T14:16:00+01:00' },
    { task:'tripletex', what:'AI accounting agent deployed på Cloud Run', why:'Agent som parser naturlig språk og utfører Tripletex API-kall', result:'Kjører live', status:'active', time:'2026-03-19T20:00:00+01:00' },
    { task:'tripletex', what:'Fjernet fresh sandbox remap', why:'register_payment_by_search ble feilaktig remappet', result:'Fikset', status:'done', time:'2026-03-20T12:35:00+01:00' },
    { task:'general', what:'GPU-trening startet på 3 GCP VM-er', why:'Parallell trening: DINOv2 + YOLOv8x multi-class', result:'Best mAP50: 0.786', status:'active', time:'2026-03-20T08:00:00+01:00' },
    { task:'astar', what:'Autopilot med 3-layer hybrid predictor', why:'KT estimator + GT lookup + adaptive priors', result:'R4=78.8 (best), R6=60.6 (falling)', status:'active', time:'2026-03-20T09:00:00+01:00' },
  ],
};
