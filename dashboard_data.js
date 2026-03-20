// dashboard_data.js — All competition state in one file
// Oppdater denne filen direkte (eller via script) for å oppdatere dashboardet.
// Dashboardet leser BARE herfra — ingen localStorage.

const COMPETITION_DATA = {

  // ═══════════════════════════════════════════
  // NORGESGRUPPEN — Object Detection
  // ═══════════════════════════════════════════
  norgesgruppen: {
    bestScore: 0.9139,
    rank: 4,
    totalTeams: 167,
    dailyQuotaUsed: 2,
    dailyQuotaMax: 6,
    inFlight: 0,
    submissions: [
      { id:1, date:'2026-03-19', time:'21:00', zip:'submission.zip', desc:'YOLOv8x multi-class, no classifier', score:0.4759, runtime:165.2, status:'completed', notes:'det~0.68, cls~0' },
      { id:2, date:'2026-03-19', time:'22:00', zip:'submission_v2.zip', desc:'YOLOv8x + WBF multi-scale', score:null, runtime:null, status:'failed', notes:'Exit code 1' },
      { id:3, date:'2026-03-20', time:'10:33', zip:'submission_v3.zip', desc:'YOLOv8x single-class + EfficientNet-B3', score:null, runtime:null, status:'skipped', notes:'Ikke submittet' },
      { id:4, date:'2026-03-20', time:'10:38', zip:'submission_20260320_103349.zip', desc:'YOLO26-x ONNX + DINOv2, score=det*cls^0.15', score:0.6740, runtime:287.9, status:'completed', notes:'+41.6% vs forrige' },
      { id:5, date:'2026-03-20', time:'11:52', zip:'submission_20260320_115200.zip', desc:'YOLO26-x ONNX + DINOv2 + multi-class YOLO hybrid', score:null, runtime:null, status:'failed', notes:'Exit code 1' },
      { id:6, date:'2026-03-20', time:'14:16', zip:'submission_20260320_141629.zip', desc:'YOLO26-x FP16 ONNX + DINOv2 only, pure det ranking', score:null, runtime:null, status:'skipped', notes:'Ikke submittet' },
      { id:7, date:'2026-03-21', time:'15:30', zip:'submission_20260321_153000.zip', desc:'YOLO26-x ONNX + DINOv2-Base v2', score:null, runtime:null, status:'skipped', notes:'Ikke submittet' },
      { id:8, date:'2026-03-20', time:'18:41', zip:'submission_20260320_182916.zip', desc:'3-modell WBF ensemble (pseudo 0.789 + fold0 0.726 + fold2 0.749) + TTA', score:0.9139, runtime:38.2, status:'completed', notes:'Ensemble er game-changer! +35.6%' },
    ],
    leaderboard: [
      { name:'Havvind', score:0.9200, subs:null },
      { name:'prompt injection 1678', score:0.9199, subs:8 },
      { name:'sf', score:0.9193, subs:5 },
      { name:'OSS', score:0.9139, subs:3, us:true },
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
    updated: '2026-03-20T18:45:00+01:00',
  },

  // ═══════════════════════════════════════════
  // GPU TRAINING STATUS
  // ═══════════════════════════════════════════
  training: {
    vms: [
      { name:'yolo26-a100', gpu:'A100', zone:'us-central1-b', task:'K-fold 0,1,2 done + fold 3 trener', detail:'Fold3 ep62, best fold2 mAP50=0.749', online:true },
      { name:'yolo26-train', gpu:'L4', zone:'eu-west4-a', task:'Pseudo-label multi-class', detail:'Epoch 132, mAP50=0.789 (best)', online:true },
      { name:'nmiai-train-fast', gpu:'L4', zone:'us-central1-a', task:'Progressive resize pretrain', detail:'Trener', online:true },
      { name:'yolo26-l4-3', gpu:'L4', zone:'eu-west4-c', task:'K-fold 1 (parallell)', detail:'Epoch 82, mAP50=0.717', online:true },
      { name:'yolo26-t4-1', gpu:'T4', zone:'us-central1-a', task:'K-fold 2 (parallell)', detail:'Epoch 91, mAP50=0.736', online:true },
      { name:'yolo26-t4-2', gpu:'T4', zone:'us-central1-a', task:'K-fold 3 (parallell)', detail:'Epoch 89, mAP50=0.693', online:true },
      { name:'classifier-train', gpu:'L4', zone:'eu-west4-a', task:'DINOv2 v2 (DEPRECATED)', detail:'Epoch 44/50, val=91.8%', online:false },
    ],
    history: [
      { time:'2026-03-20T08:30:00Z', best:0.649, note:'First multi-class epoch' },
      { time:'2026-03-20T09:00:00Z', best:0.735, note:'Epoch 2' },
      { time:'2026-03-20T09:30:00Z', best:0.744, note:'Epoch 3' },
      { time:'2026-03-20T10:00:00Z', best:0.761, note:'Epoch 5' },
      { time:'2026-03-20T10:30:00Z', best:0.772, note:'Epoch 8' },
      { time:'2026-03-20T11:00:00Z', best:0.783, note:'Epoch 10' },
      { time:'2026-03-20T11:30:00Z', best:0.786, note:'Epoch 11' },
      { time:'2026-03-20T14:00:00Z', best:0.789, note:'Pseudo-label model best' },
      { time:'2026-03-20T16:00:00Z', best:0.789, note:'K-fold training started (4 folds)' },
      { time:'2026-03-20T17:30:00Z', best:0.789, note:'Fold0=0.726, Fold2=0.749 done' },
      { time:'2026-03-20T18:41:00Z', best:0.9139, note:'3-modell WBF ensemble = 0.9139!' },
    ],
    bestMap50: 0.789,
    target: 0.920,
  },

  // ═══════════════════════════════════════════
  // HISTORIKK — Hva vi har gjort og hvorfor
  // ═══════════════════════════════════════════
  history: [
    { task:'norgesgruppen', what:'YOLOv8x multi-class detection', why:'Raskeste baseline for deteksjon', result:'det mAP~0.68, cls mAP~0. Score: 0.4759', status:'done', time:'2026-03-19T21:00:00+01:00' },
    { task:'norgesgruppen', what:'WBF multi-scale ensemble', why:'Booste deteksjon med TTA', result:'Failed — exit code 1', status:'abandoned', time:'2026-03-19T22:00:00+01:00' },
    { task:'norgesgruppen', what:'YOLO26-x ONNX + DINOv2 classifier', why:'YOLO26 SOTA + DINOv2 robust embeddings', result:'Score: 0.6740 (+41.6%)', status:'done', time:'2026-03-20T10:38:00+01:00' },
    { task:'norgesgruppen', what:'Hybrid multi-class YOLO + DINOv2', why:'Kombinere single-class + multi-class', result:'Failed — exit code 1', status:'abandoned', time:'2026-03-20T11:52:00+01:00' },
    { task:'norgesgruppen', what:'K-fold trening startet (5 folds)', why:'Diverse modeller for ensemble, bedre generalisering', result:'Fold0=0.726, Fold1=0.718, Fold2=0.749', status:'done', time:'2026-03-20T14:00:00+01:00' },
    { task:'norgesgruppen', what:'Pseudo-label multi-class modell', why:'Semi-supervised: bruk prediksjoner som labels for uannoterte bilder', result:'mAP50=0.789 — beste enkeltmodell', status:'done', time:'2026-03-20T14:00:00+01:00' },
    { task:'norgesgruppen', what:'3-modell WBF ensemble + TTA', why:'Ensemble av pseudo(0.789)+fold0(0.726)+fold2(0.749) med WBF fusion', result:'Score: 0.9139! +35.6%, ~4. plass, gap 0.006 til topp', status:'done', time:'2026-03-20T18:41:00+01:00' },
    { task:'norgesgruppen', what:'Multi-class YOLO slår two-stage', why:'DINOv2 classifier var flaskehals. Multi-class YOLO gir det+cls i ett pass', result:'Hele arkitekturen endret fra two-stage til ensemble', status:'done', time:'2026-03-20T18:00:00+01:00' },
    { task:'tripletex', what:'AI accounting agent deployed på Cloud Run', why:'Agent parser naturlig språk → Tripletex API', result:'Kjører live, prosesserer oppgaver', status:'active', time:'2026-03-19T20:00:00+01:00' },
    { task:'tripletex', what:'Fjernet fresh sandbox remap', why:'register_payment_by_search feilaktig remappet', result:'Fikset', status:'done', time:'2026-03-20T12:35:00+01:00' },
    { task:'general', what:'GPU-trening fleet: 7 VM-er på GCP', why:'Parallell K-fold + pseudo-label + progressive resize', result:'Best single: 0.789, best ensemble: 0.9139', status:'active', time:'2026-03-20T08:00:00+01:00' },
    { task:'astar', what:'Autopilot med 3-layer hybrid predictor', why:'KT estimator + GT lookup + adaptive priors', result:'R4=78.8 (best), R6=60.6 (falling)', status:'active', time:'2026-03-20T09:00:00+01:00' },
    { task:'norgesgruppen', what:'Neste: bedre ensemble med fold3 + tune WBF', why:'Fold3 trener, pseudo trener videre. Tune iou_thr, conf, TTA scales', result:'Venter ~30 min', status:'planned', time:'2026-03-20T18:45:00+01:00' },
  ],

  // ═══════════════════════════════════════════
  // SCORE HISTORIKK — Kort oversikt
  // ═══════════════════════════════════════════
  scoreHistory: [
    { sub:1, score:0.4759, method:'YOLOv8x single model' },
    { sub:4, score:0.6740, method:'YOLO26-x + DINOv2 (timeout)' },
    { sub:8, score:0.9139, method:'3-modell WBF ensemble + TTA' },
  ],

  // ═══════════════════════════════════════════
  // ONNX MODELLER TILGJENGELIG
  // ═══════════════════════════════════════════
  models: [
    { name:'pseudo_best.onnx', map50:0.789, size:'109 MB', source:'Pseudo-label (L4)', status:'downloaded' },
    { name:'fold2_best.onnx', map50:0.749, size:'108 MB', source:'K-fold 2 (A100)', status:'downloaded' },
    { name:'fold0_best.onnx', map50:0.726, size:'109 MB', source:'K-fold 0 (A100)', status:'downloaded' },
    { name:'fold1_best.onnx', map50:0.718, size:'113 MB', source:'K-fold 1 (A100)', status:'downloaded' },
    { name:'fold3', map50:0.678, size:'~108 MB', source:'K-fold 3 (A100)', status:'training' },
  ],
};
