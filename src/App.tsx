import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { BrowserRouter, Link, Route, Routes, useNavigate } from "react-router-dom";
import * as handpose from "@tensorflow-models/handpose";
import "@tensorflow/tfjs";
import * as tf from "@tensorflow/tfjs";

const AppContext = createContext(null);

const uid = () => Math.random().toString(36).slice(2, 10);

const createEmptyClass = (index: number) => ({
  id: uid(),
  name: `Class ${index + 1}`,
  samples: [],
});

function clamp01(n) {
  return Math.max(0, Math.min(1, n));
}

function safeNumber(v: string, fallback: number) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function flattenLandmarks(landmarks) {
  return landmarks.flatMap(([x, y, z]) => [x, y, z]);
}

function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;

  const wrist = landmarks[0];
  let maxDist = 0;

  for (const [x, y, z] of landmarks) {
    const dx = x - wrist[0];
    const dy = y - wrist[1];
    const dz = (z || 0) - (wrist[2] || 0);
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > maxDist) maxDist = dist;
  }

  const scale = maxDist || 1;
  return landmarks.map(([x, y, z]) => [
    (x - wrist[0]) / scale,
    (y - wrist[1]) / scale,
    ((z || 0) - (wrist[2] || 0)) / scale,
  ]);
}

function vectorizeLandmarks(landmarks) {
  const normalized = normalizeLandmarks(landmarks);
  if (!normalized) return null;
  return flattenLandmarks(normalized);
}

function captureVideoFrame(video) {
  if (!video) return null;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

function argMax(values) {
  let maxIdx = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[maxIdx]) maxIdx = i;
  }
  return maxIdx;
}

function extractAverageColor(video) {
  if (!video) return null;
  const canvas = document.createElement("canvas");
  const size = 20; // Analyze center 20x20 pixels
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const sx = (video.videoWidth - size) / 2;
  const sy = (video.videoHeight - size) / 2;
  ctx.drawImage(video, sx, sy, size, size, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  let r = 0, g = 0, b = 0;
  for (let i = 0; i < data.length; i += 4) {
    r += data[i]; g += data[i + 1]; b += data[i + 2];
  }
  const count = data.length / 4;
  return [r / count / 255, g / count / 255, b / count / 255]; // Normalize 0-1
}

function useApp() {
  return useContext(AppContext);
}

function AppProvider({ children }) {
  const handposeModelRef = useRef<handpose.HandPose | null>(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelError, setModelError] = useState("");

  // ✅ FIX: cameraOn state moved inside AppProvider (was outside component before)
  const [cameraOn, setCameraOn] = useState(false);

  const [classes, setClasses] = useState([]);
  const [datasetVersion, setDatasetVersion] = useState(0);

  const [advancedMode, setAdvancedMode] = useState(false);
  const [epochs, setEpochs] = useState(20);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);

  const [trainingStatus, setTrainingStatus] = useState("Not trained");
  const [isTraining, setIsTraining] = useState(false);
  const [trainedModel, setTrainedModel] = useState(null);
  const [trainedRevision, setTrainedRevision] = useState(-1);
  const [lastTrainSummary, setLastTrainSummary] = useState(null);

  const markDatasetDirty = useCallback(() => {
    setDatasetVersion((v) => v + 1);
    setTrainedModel(null);
    setTrainedRevision(-1);
    setLastTrainSummary(null);
    setTrainingStatus("Dataset changed — retrain required");
  }, []);

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        setModelLoading(true);
        const model = await handpose.load();
        if (!mounted) return;
        handposeModelRef.current = model;
        setModelError("");
      } catch (err) {
        if (!mounted) return;
        setModelError(err?.message || "Failed to load handpose model");
      } finally {
        if (mounted) setModelLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  const addClass = useCallback(() => {
    setClasses((prev) => [...prev, createEmptyClass(prev.length)]);
    markDatasetDirty();
  }, [markDatasetDirty]);

  const renameClass = useCallback((classId, value) => {
    setClasses((prev) =>
      prev.map((cls) => (cls.id === classId ? { ...cls, name: value } : cls))
    );
    markDatasetDirty();
  }, [markDatasetDirty]);

  const deleteClass = useCallback((classId) => {
    setClasses((prev) => prev.filter((cls) => cls.id !== classId));
    markDatasetDirty();
  }, [markDatasetDirty]);

  const addSampleToClass = useCallback((classId, sample) => {
    setClasses((prev) =>
      prev.map((cls) =>
        cls.id === classId
          ? { ...cls, samples: [sample, ...cls.samples] }
          : cls
      )
    );
    markDatasetDirty();
  }, [markDatasetDirty]);

  const deleteSample = useCallback((classId, sampleId) => {
    setClasses((prev) =>
      prev.map((cls) =>
        cls.id === classId
          ? { ...cls, samples: cls.samples.filter((s) => s.id !== sampleId) }
          : cls
      )
    );
    markDatasetDirty();
  }, [markDatasetDirty]);

  const trainModel = useCallback(async () => {
    if (isTraining) return;

    const totalSamples = classes.reduce((sum, cls) => sum + cls.samples.length, 0);
    if (!classes.length || totalSamples === 0) {
      setTrainingStatus("Add at least one class and one sample first");
      return;
    }

    setIsTraining(true);
    setTrainingStatus(advancedMode ? "Training Neural Network..." : "Preparing KNN model...");

    try {
      // Filter empty classes
      const activeClasses = classes.filter(c => c.samples.length > 0);
      const labelMap = activeClasses.map((cls) => ({ id: cls.id, name: cls.name }));
      
      // --- Prepare Hand Model Data (Feature len 63) ---
      let handModel = null;
      let colorModel = null;
      let totalHandSamples = 0;
      let totalColorSamples = 0;

      if (!advancedMode) {
        // --- KNN MODE ---
        const handKnnData = [];
        const colorKnnData = [];

        for (const cls of activeClasses) {
          for (const sample of cls.samples) {
            if (sample.features?.length === 63) {
              handKnnData.push({ classId: cls.id, className: cls.name, features: sample.features });
            } else if (sample.features?.length === 3) {
              colorKnnData.push({ classId: cls.id, className: cls.name, features: sample.features });
            }
          }
        }

        if (handKnnData.length > 0) {
          handModel = { type: 'knn', data: handKnnData };
          totalHandSamples = handKnnData.length;
        }
        if (colorKnnData.length > 0) {
          colorModel = { type: 'knn', data: colorKnnData };
          totalColorSamples = colorKnnData.length;
        }

      } else {
        // --- NN MODE ---
        
        // Helper to train a model
        const trainNN = async (inputSize, getSamples) => {
            const xs = [];
            const ys = [];
            activeClasses.forEach((cls, idx) => {
                cls.samples.forEach(s => {
                    if (s.features?.length === inputSize) {
                        xs.push(s.features);
                        ys.push(idx);
                    }
                });
            });
            
            if (xs.length === 0) return { model: null, count: 0 };

            const xTensor = tf.tensor2d(xs);
            const yTensor = tf.oneHot(tf.tensor1d(ys, "int32"), activeClasses.length);
            
            const model = tf.sequential();
            // Larger architecture for hands (63 inputs), smaller for colors (3 inputs)
            if (inputSize === 63) {
                model.add(tf.layers.dense({ inputShape: [63], units: 128, activation: "relu" }));
                model.add(tf.layers.dense({ units: 64, activation: "relu" }));
            } else {
                model.add(tf.layers.dense({ inputShape: [3], units: 16, activation: "relu" }));
            }
            model.add(tf.layers.dense({ units: activeClasses.length, activation: "softmax" }));

            model.compile({
                optimizer: tf.train.adam(learningRate),
                loss: "categoricalCrossentropy",
                metrics: ["accuracy"]
            });

            await model.fit(xTensor, yTensor, {
                epochs,
                batchSize,
                shuffle: true,
                callbacks: {
                   onEpochEnd: (epoch, logs) => {
                       // Only show logs for hand training to avoid spam, or last one
                       if (inputSize === 63) {
                         const loss = logs?.loss != null ? logs.loss.toFixed(4) : "-";
                         setTrainingStatus(`Epoch ${epoch + 1}/${epochs} | loss: ${loss}`);
                       }
                   }
                }
            });
            
            xTensor.dispose();
            yTensor.dispose();
            return { model, count: xs.length };
        };

        // Train Hand Model
        const handRes = await trainNN(63);
        if (handRes.model) {
            handModel = { type: 'nn', model: handRes.model };
            totalHandSamples = handRes.count;
        }

        // Train Color Model
        const colorRes = await trainNN(3);
        if (colorRes.model) {
            colorModel = { type: 'nn', model: colorRes.model };
            totalColorSamples = colorRes.count;
        }
      }

      if (!handModel && !colorModel) {
        setTrainingStatus("No valid samples (Hand or Color) found");
        return;
      }

      setTrainedModel({
        mode: advancedMode ? "advanced" : "basic",
        labelMap,
        hand: handModel,
        color: colorModel
      });
      setTrainedRevision(datasetVersion);
      setLastTrainSummary({
        mode: advancedMode ? "advanced" : "basic",
        samples: totalHandSamples + totalColorSamples,
        classes: activeClasses.length,
      });
      setTrainingStatus(`Ready ✅ | Hand: ${totalHandSamples}, Color: ${totalColorSamples} samples`);
      
    } catch (err) {
      setTrainingStatus(err?.message || "Training failed");
    } finally {
      setIsTraining(false);
    }
  }, [advancedMode, batchSize, classes, datasetVersion, epochs, isTraining, learningRate]);

  const resetAll = useCallback(() => {
    setClasses([]);
    setDatasetVersion(0);
    setTrainedModel(null);
    setTrainedRevision(-1);
    setLastTrainSummary(null);
    setTrainingStatus("Not trained");
    setCameraOn(false);
  }, []);

  const value = useMemo(
    () => ({
      handposeModelRef,
      modelLoading,
      modelError,
      classes,
      datasetVersion,
      advancedMode,
      setAdvancedMode,
      epochs,
      setEpochs,
      batchSize,
      setBatchSize,
      learningRate,
      setLearningRate,
      trainingStatus,
      setTrainingStatus,
      isTraining,
      trainedModel,
      trainedRevision,
      lastTrainSummary,
      addClass,
      renameClass,
      deleteClass,
      addSampleToClass,
      deleteSample,
      trainModel,
      resetAll,
      markDatasetDirty,
      setClasses,
      // ✅ FIX: expose cameraOn and setCameraOn via context
      cameraOn,
      setCameraOn,
    }),
    [
      addClass,
      addSampleToClass,
      advancedMode,
      batchSize,
      cameraOn,
      classes,
      datasetVersion,
      deleteClass,
      deleteSample,
      epochs,
      handposeModelRef,
      isTraining,
      learningRate,
      lastTrainSummary,
      markDatasetDirty,
      modelError,
      modelLoading,
      renameClass,
      resetAll,
      trainedModel,
      trainedRevision,
      trainModel,
      trainingStatus,
    ]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

function useHandTracking(videoEl, onResults, enabled = true) {
  const { handposeModelRef, modelLoading } = useApp();
  const onResultsRef = useRef(onResults);
  const streamRef = useRef(null);

  useEffect(() => {
    onResultsRef.current = onResults;
  }, [onResults]);

  useEffect(() => {
    if (!enabled) {
      // ✅ FIX: stop camera stream when disabled
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (videoEl) {
        videoEl.srcObject = null;
      }
      return;
    }

    if (modelLoading || !handposeModelRef.current || !videoEl) return;

    let active = true;
    let rafId = 0;

    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });

        streamRef.current = stream;

        if (!active || !videoEl) return;

        videoEl.srcObject = stream;
        await videoEl.play();

        const detect = async () => {
          if (!active || !videoEl || !handposeModelRef.current) return;

          try {
            if (videoEl.readyState >= 2) {
              const predictions = await handposeModelRef.current.estimateHands(videoEl, true);
              onResultsRef.current(predictions || []);
            }
          } catch {
            onResultsRef.current([]);
          }

          rafId = requestAnimationFrame(detect);
        };

        detect();
      } catch (err) {
        onResultsRef.current({ error: err?.message || "Camera access failed" });
      }
    };

    start();

    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, [handposeModelRef, modelLoading, videoEl, enabled]);
}

function usePredictionEngine() {
  const { trainedModel } = useApp();

  const predict = useCallback(
    (landmarks, videoElement) => {
      if (!trainedModel) return { className: "", confidence: 0, type: "none" };

      // DECIDE: Hand or Color?
      // If landmarks exist -> Hand Model. Else if videoElement -> Color Model.
      let features = null;
      let activeModel = null;
      let type = "none";

      if (landmarks && trainedModel.hand) {
          features = vectorizeLandmarks(landmarks); // returns 63 length array
          activeModel = trainedModel.hand;
          type = "hand";
      } else if (videoElement && trainedModel.color) {
          features = extractAverageColor(videoElement); // returns 3 length array
          activeModel = trainedModel.color;
          type = "color";
      }

      if (!features || !activeModel) return { className: "", confidence: 0, type: "none" };

      // --- KNN PREDICTION ---
      if (activeModel.type === 'knn') {
        const k = Math.min(3, activeModel.data.length);
        const scored = activeModel.data
          .map((row) => {
            const distance = Math.sqrt(
              row.features.reduce((sum, val, i) => sum + (val - features[i]) ** 2, 0)
            );
            return { classId: row.classId, className: row.className, distance };
          })
          .sort((a, b) => a.distance - b.distance).slice(0, k);

        const votes = {};
        scored.forEach((item) => {
          if (!votes[item.classId]) votes[item.classId] = { className: item.className, count: 0, dist: 0 };
          votes[item.classId].count += 1;
          votes[item.classId].dist += item.distance;
        });
        const best = Object.entries(votes).sort((a, b) => {
          if (b[1].count !== a[1].count) return b[1].count - a[1].count;
          return a[1].dist - b[1].dist;
        })[0];

        const confidence = best ? (best[1].count / k) * 100 : 0;
        return { className: best ? best[1].className : "", confidence: Math.round(confidence) };
      }
      
      // --- NN PREDICTION ---
      const input = tf.tensor2d([features]);
      const output = activeModel.model.predict(input);
      const probs = output.dataSync();
      const idx = argMax(probs);
      const confidence = Math.round((probs[idx] || 0) * 100);
      input.dispose();
      output.dispose();

      return {
        className: trainedModel.labelMap[idx]?.name || "",
        confidence,
        type 
      };
    },
    [trainedModel]
  );

  return { predict };
}

function Header() {
  const { modelLoading, modelError, trainedModel, trainingStatus, lastTrainSummary } = useApp();
  const ready = Boolean(trainedModel);

  return (
    <header style={styles.header}>
      <div>
        <div style={styles.brand}>Hand Gesture Recognition</div>
        <div style={styles.subBrand}>Dataset creation → training → real-time testing</div>
      </div>

      <div style={styles.headerPills}>
        <span style={{ ...styles.pill, ...(modelLoading ? styles.pillWarning : styles.pillSuccess) }}>
          {modelLoading ? "Loading hand model..." : modelError ? "Model load failed" : "Hand model ready"}
        </span>
        <span style={{ ...styles.pill, ...(ready ? styles.pillSuccess : styles.pillMuted) }}>
          {ready ? "Trained model ready" : "Not trained"}
        </span>
        <span style={{ ...styles.pill, ...styles.pillMuted }}>{trainingStatus}</span>
        {lastTrainSummary ? (
          <span style={{ ...styles.pill, ...styles.pillMuted }}>
            {lastTrainSummary.mode.toUpperCase()} · {lastTrainSummary.samples} samples
          </span>
        ) : null}
      </div>
    </header>
  );
}

function SampleCard({ sample, onDelete }) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      style={{
        ...styles.sampleCard,
        transform: isHovered ? "translateY(-4px)" : "translateY(0)",
        boxShadow: isHovered ? "0 8px 20px rgba(0, 0, 0, 0.15)" : "0 2px 8px rgba(0, 0, 0, 0.05)",
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <img src={sample.imageUrl} alt="sample" style={styles.sampleImage} />
      <div style={styles.sampleMetaRow}>
        <span style={styles.sampleBadge}>{sample.source}</span>
        <button
          onClick={onDelete}
          style={{
            ...styles.iconButton,
            transform: isHovered ? "scale(1.1)" : "scale(1)",
          }}
          title="Delete sample"
        >
          ✕
        </button>
      </div>
      <div style={styles.sampleHint}>
        {sample.features?.length === 63 ? "Hand (63D)" : sample.features?.length === 3 ? "Color (RGB)" : "No features"}
      </div>
    </div>
  );
}

function TrainingPage() {
  const navigate = useNavigate();
  const {
    classes,
    addClass,
    renameClass,
    deleteClass,
    deleteSample,
    addSampleToClass,
    advancedMode,
    setAdvancedMode,
    epochs,
    setEpochs,
    batchSize,
    setBatchSize,
    learningRate,
    setLearningRate,
    trainModel,
    isTraining,
    trainedModel,
    trainingStatus,
    modelLoading,
    modelError,
    trainedRevision,
    datasetVersion,
    resetAll,
    handposeModelRef,
    setTrainingStatus,
    // ✅ FIX: get cameraOn and setCameraOn from context
    cameraOn,
    setCameraOn,
  } = useApp();

  const [videoEl, setVideoEl] = useState(null);
  const latestHandsRef = useRef([]);
  const liveHandStateRef = useRef({ lastSeen: 0, label: "No hand detected" });
  const [, forceRerender] = useState(0);

  useHandTracking(videoEl, (predictions) => {
    if (predictions?.error) {
      setTrainingStatus(predictions.error);
      return;
    }

    latestHandsRef.current = predictions || [];
    if (predictions && predictions.length) {
      liveHandStateRef.current = {
        lastSeen: Date.now(),
        label: `Hand detected (${predictions.length})`,
      };
    } else if (Date.now() - liveHandStateRef.current.lastSeen > 600) {
      liveHandStateRef.current = { lastSeen: Date.now(), label: "No hand detected" };
    }

    forceRerender((n) => n + 1);
  }, cameraOn); // ✅ FIX: pass cameraOn correctly

  const canCollect = !modelLoading && !modelError && handposeModelRef.current;

  const collectFromWebcam = async (classId) => {
    const prediction = latestHandsRef.current?.[0];
    if (!prediction?.landmarks) {
      setTrainingStatus("Show a hand in the webcam first");
      return;
    }

    const features = vectorizeLandmarks(prediction.landmarks);
    if (!features || features.length !== 63) {
      setTrainingStatus("Could not extract a valid 63-length feature vector");
      return;
    }

    const imageUrl = captureVideoFrame(videoEl);
    addSampleToClass(classId, {
      id: uid(),
      source: "webcam",
      imageUrl,
      landmarks: prediction.landmarks,
      features,
      createdAt: Date.now(),
    });

    setTrainingStatus("Webcam sample added ✅");
  };

  // NEW: Collect color
  const collectColor = async (classId) => {
      if(!videoEl) return;
      const features = extractAverageColor(videoEl);
      // We reuse captureVideoFrame for the thumbnail, but we train on the extracted average color
      const imageUrl = captureVideoFrame(videoEl);
      
      addSampleToClass(classId, {
          id: uid(), source: "color", imageUrl, features, createdAt: Date.now()
      });
      setTrainingStatus("Color sample added ✅");
  };

  const handleUpload = async (event, classId) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    if (!handposeModelRef.current) {
      setTrainingStatus("Handpose model is still loading");
      return;
    }

    setTrainingStatus(`Processing ${files.length} image(s)...`);

    for (const file of files) {
      const imageUrl = URL.createObjectURL(file);
      const img = new Image();
      img.src = imageUrl;
      img.decoding = "async";

      await new Promise((resolve) => {
        img.onload = resolve;
        img.onerror = resolve;
      });

      try {
        const preds = await handposeModelRef.current.estimateHands(img, true);
        const hand = preds?.[0];
        const features = hand?.landmarks ? vectorizeLandmarks(hand.landmarks) : null;

        if (hand?.landmarks && features && features.length === 63) {
          addSampleToClass(classId, {
            id: uid(),
            source: "upload",
            imageUrl,
            landmarks: hand.landmarks,
            features,
            createdAt: Date.now(),
          });
        } else {
          URL.revokeObjectURL(imageUrl);
        }
      } catch {
        URL.revokeObjectURL(imageUrl);
      }
    }

    event.target.value = "";
    setTrainingStatus("Upload processed ✅");
  };

  const trainingDisabled = isTraining || classes.length === 0;
  const testReady = Boolean(trainedModel) && trainedRevision === datasetVersion;

  return (
    <div style={styles.pageShell}>
      <Header />

      <main style={styles.grid3}>
        <section style={styles.panel}>
          <div style={styles.panelHeaderRow}>
            <h2 style={styles.panelTitle}>Class & Dataset Management</h2>
            <button
              onClick={addClass}
              style={{ ...styles.primaryButton }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "scale(1.05)";
                e.currentTarget.style.boxShadow = "0 8px 24px rgba(102, 126, 234, 0.5)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "scale(1)";
                e.currentTarget.style.boxShadow = "0 4px 14px rgba(102, 126, 234, 0.4)";
              }}
            >
              + Add Class
            </button>
          </div>

          {/* ✅ Camera toggle row */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span style={{ fontWeight: 600, color: "#0f172a" }}>Camera</span>
            <button
              onClick={() => setCameraOn((prev) => !prev)}
              style={{
                ...styles.primaryButton,
                background: cameraOn
                  ? "linear-gradient(135deg, #ef4444, #dc2626)"
                  : "linear-gradient(135deg, #10b981, #059669)",
                boxShadow: cameraOn
                  ? "0 4px 14px rgba(239, 68, 68, 0.4)"
                  : "0 4px 14px rgba(16, 185, 129, 0.4)",
              }}
            >
              {cameraOn ? "Stop Camera" : "Start Camera"}
            </button>
          </div>

          <div style={styles.trainVideoWrap}>
            <video ref={setVideoEl} style={styles.trainVideo} muted playsInline autoPlay />
            <div style={styles.cameraOverlay}>{liveHandStateRef.current.label}</div>
            {/* Center target for color sampling */}
            <div style={{
                position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                width: 24, height: 24, border: '2px solid rgba(255,255,255,0.8)', borderRadius: 4, pointerEvents: 'none'
            }} />
          </div>

          <div style={styles.classList}>
            {classes.length === 0 ? (
              <div style={styles.emptyState}>No classes yet. Add one to start collecting samples.</div>
            ) : null}

            {classes.map((cls) => (
              <div key={cls.id} style={styles.classCard}>
                <div style={styles.classCardTop}>
                  <input
                    value={cls.name}
                    onChange={(e) => renameClass(cls.id, e.target.value)}
                    placeholder="Class name"
                    style={styles.classInput}
                    onFocus={(e) => {
                      e.target.style.borderColor = "#667eea";
                      e.target.style.boxShadow = "0 0 0 3px rgba(102, 126, 234, 0.1)";
                    }}
                    onBlur={(e) => {
                      e.target.style.borderColor = "#e2e8f0";
                      e.target.style.boxShadow = "none";
                    }}
                  />
                  <button
                    onClick={() => deleteClass(cls.id)}
                    style={styles.dangerButton}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = "#fee2e2";
                      e.currentTarget.style.borderColor = "#ef4444";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "#fef2f2";
                      e.currentTarget.style.borderColor = "#fecaca";
                    }}
                  >
                    Delete
                  </button>
                </div>

                <div style={styles.classMetaRow}>
                  <span style={styles.countChip}>{cls.samples.length} samples</span>
                  <span style={styles.countChipMuted}>Dataset v{datasetVersion}</span>
                </div>

                <div style={styles.collectRow}>
                  <button
                    onClick={() => collectFromWebcam(cls.id)}
                    style={{ 
                      ...styles.secondaryButton,
                      opacity: canCollect ? 1 : 0.5,
                      cursor: canCollect ? "pointer" : "not-allowed",
                    }}
                    disabled={!canCollect}
                    onMouseEnter={(e) => {
                      if (canCollect) {
                        e.currentTarget.style.background = "#f8fafc";
                        e.currentTarget.style.borderColor = "#667eea";
                      }
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "#ffffff";
                      e.currentTarget.style.borderColor = "#e2e8f0";
                    }}
                    title="Detect hand and add as sample"
                  >
                    Capture Hand
                  </button>
                  <button
                    onClick={() => collectColor(cls.id)}
                    style={styles.secondaryButton}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "#f8fafc"; e.currentTarget.style.borderColor = "#667eea"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "#ffffff"; e.currentTarget.style.borderColor = "#e2e8f0"; }}
                    title="Capture center color as sample"
                  >
                    Capture Color
                  </button>

                  <label
                    style={styles.secondaryButton}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = "#f8fafc";
                      e.currentTarget.style.borderColor = "#667eea";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "#ffffff";
                      e.currentTarget.style.borderColor = "#e2e8f0";
                    }}
                  >
                    Upload Images
                    <input type="file" multiple accept="image/*" onChange={(e) => handleUpload(e, cls.id)} hidden />
                  </label>
                </div>

                <div style={styles.sampleScroller}>
                  {cls.samples.length === 0 ? (
                    <div style={styles.samplePlaceholder}>No samples yet.</div>
                  ) : (
                    cls.samples.map((sample) => (
                      <SampleCard
                        key={sample.id}
                        sample={sample}
                        onDelete={() => deleteSample(cls.id, sample.id)}
                      />
                    ))
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section style={styles.panel}>
          <div style={styles.panelHeaderRow}>
            <h2 style={styles.panelTitle}>Model Training</h2>
            <div style={styles.modeToggleWrap}>
              <button
                style={{ ...styles.toggleButton, ...(advancedMode ? {} : styles.toggleActive) }}
                onClick={() => setAdvancedMode(false)}
              >
                Basic KNN
              </button>
              <button
                style={{ ...styles.toggleButton, ...(advancedMode ? styles.toggleActive : {}) }}
                onClick={() => setAdvancedMode(true)}
              >
                Advanced NN
              </button>
            </div>
          </div>

          <div style={styles.subText}>
            Basic mode is fast and lightweight. Advanced mode trains a TensorFlow.js neural network with configurable hyperparameters.
          </div>

          {advancedMode ? (
            <div style={styles.paramGrid}>
              <label style={styles.paramBox}>
                <span>Epochs</span>
                <input
                  type="number"
                  min="1"
                  value={epochs}
                  onChange={(e) => setEpochs(safeNumber(e.target.value, 20))}
                  style={styles.numberInput}
                  onFocus={(e) => {
                    e.target.style.borderColor = "#667eea";
                    e.target.style.boxShadow = "0 0 0 3px rgba(102, 126, 234, 0.1)";
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = "#e2e8f0";
                    e.target.style.boxShadow = "none";
                  }}
                />
              </label>
              <label style={styles.paramBox}>
                <span>Batch Size</span>
                <input
                  type="number"
                  min="1"
                  value={batchSize}
                  onChange={(e) => setBatchSize(safeNumber(e.target.value, 16))}
                  style={styles.numberInput}
                  onFocus={(e) => {
                    e.target.style.borderColor = "#667eea";
                    e.target.style.boxShadow = "0 0 0 3px rgba(102, 126, 234, 0.1)";
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = "#e2e8f0";
                    e.target.style.boxShadow = "none";
                  }}
                />
              </label>
              <label style={styles.paramBox}>
                <span>Learning Rate</span>
                <input
                  type="number"
                  min="0.0001"
                  step="0.0001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(safeNumber(e.target.value, 0.001))}
                  style={styles.numberInput}
                  onFocus={(e) => {
                    e.target.style.borderColor = "#667eea";
                    e.target.style.boxShadow = "0 0 0 3px rgba(102, 126, 234, 0.1)";
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = "#e2e8f0";
                    e.target.style.boxShadow = "none";
                  }}
                />
              </label>
            </div>
          ) : (
            <div style={styles.basicInfoBox}>
              <strong>KNN</strong> compares the current 63-length feature vector against stored samples.
            </div>
          )}

          <div style={styles.trainingCard}>
            <div style={styles.trainingStatusLabel}>Status</div>
            <div style={styles.trainingStatusText}>{trainingStatus}</div>
            <div style={styles.smallNote}>
              Training is required again whenever classes or samples change.
            </div>
            <div style={styles.trainActions}>
              <button
                onClick={trainModel}
                disabled={trainingDisabled}
                style={{
                  ...styles.primaryButton,
                  opacity: trainingDisabled ? 0.5 : 1,
                  cursor: trainingDisabled ? "not-allowed" : "pointer",
                }}
                onMouseEnter={(e) => {
                  if (!trainingDisabled) {
                    e.currentTarget.style.transform = "scale(1.05)";
                    e.currentTarget.style.boxShadow = "0 8px 24px rgba(102, 126, 234, 0.5)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.style.boxShadow = "0 4px 14px rgba(102, 126, 234, 0.4)";
                }}
              >
                {isTraining ? "Training..." : "Train Model"}
              </button>
              <button
                onClick={resetAll}
                style={styles.ghostButton}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "#f8fafc";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                Reset All
              </button>
            </div>
          </div>
        </section>

        <section style={styles.panel}>
          <h2 style={styles.panelTitle}>Testing Access</h2>
          <div style={styles.subText}>The testing page stays locked until the latest training is complete.</div>

          <div style={styles.readyCard}>
            <div style={styles.readyBadge(testReady)}>
              {testReady ? "Model ready" : "Locked"}
            </div>
            <p style={styles.readyText}>
              {testReady
                ? "You can now open the real-time testing page and start predicting gestures."
                : "Train the model after your last dataset change to unlock testing."}
            </p>

            <button
              onClick={() => navigate("/test")}
              disabled={!testReady}
              style={{
                ...styles.primaryButton,
                width: "100%",
                opacity: testReady ? 1 : 0.45,
                cursor: testReady ? "pointer" : "not-allowed",
              }}
              onMouseEnter={(e) => {
                if (testReady) {
                  e.currentTarget.style.transform = "scale(1.02)";
                  e.currentTarget.style.boxShadow = "0 8px 24px rgba(102, 126, 234, 0.5)";
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "scale(1)";
                e.currentTarget.style.boxShadow = "0 4px 14px rgba(102, 126, 234, 0.4)";
              }}
            >
              Open Testing Page
            </button>
          </div>

          <div style={styles.readOnlyStats}>
            <div style={styles.statRow}>
              <span>Classes</span>
              <strong>{classes.length}</strong>
            </div>
            <div style={styles.statRow}>
              <span>Samples</span>
              <strong>{classes.reduce((sum, cls) => sum + cls.samples.length, 0)}</strong>
            </div>
            <div style={styles.statRow}>
              <span>Mode</span>
              <strong>{advancedMode ? "Advanced" : "Basic"}</strong>
            </div>
          </div>

          <div style={styles.linkRow}>
            <Link
              to="/test"
              style={styles.linkLike}
              onMouseEnter={(e) => {
                e.currentTarget.style.textDecoration = "underline";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.textDecoration = "none";
              }}
            >
              Go to testing page →
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}

function TestPage() {
  const navigate = useNavigate();
  const { trainedModel, trainedRevision, datasetVersion, classes } = useApp();
  const [videoEl, setVideoEl] = useState(null);
  const currentHandsRef = useRef([]);
  const [prediction, setPrediction] = useState("Waiting for hand...");
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState("Starting camera...");
  const [cameraReady, setCameraReady] = useState(false);

  const { predict } = usePredictionEngine();

  // ✅ FIX: TestPage always enables camera (true), no dependency on training cameraOn
  useHandTracking(videoEl, (predictions) => {
    if (predictions?.error) {
      setStatus(predictions.error);
      return;
    }

    currentHandsRef.current = predictions || [];
    setCameraReady(true);

    // Check for hand first
    let hand = predictions?.[0];
    
    // If fallback logic is desired: "either hand detected, if not found use color"
    // We pass landmarks if found, otherwise pass videoEl for color extraction
    const result = predict(hand ? hand.landmarks : null, videoEl);

    if (!result.className) {
        setPrediction(hand ? "Unknown Gesture" : "Unknown Color");
    }
    
    if (result.className) {
        setPrediction(result.className);
    }
    
    setConfidence(result.confidence || 0);
    setStatus(hand ? "Hand Detected" : "Color Detection (Fallback)");

  }, true); // ✅ always on for test page

  const ready = Boolean(trainedModel) && trainedRevision === datasetVersion;
  const sampleCount = classes.reduce((sum, cls) => sum + cls.samples.length, 0);

  return (
    <div style={styles.testPageShell}>
      <Header />

      {!ready ? (
        <div style={styles.blockerOverlay}>
          <div style={styles.blockerCard}>
            <h2 style={{ marginTop: 0, fontSize: 24, fontWeight: 700 }}>Testing is locked</h2>
            <p style={{ marginBottom: 24, fontSize: 15, color: "#64748b", lineHeight: 1.6 }}>
              Train the model after the latest dataset changes to enable real-time prediction.
            </p>
            <button
              onClick={() => navigate("/")}
              style={styles.primaryButton}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "scale(1.05)";
                e.currentTarget.style.boxShadow = "0 8px 24px rgba(102, 126, 234, 0.5)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "scale(1)";
                e.currentTarget.style.boxShadow = "0 4px 14px rgba(102, 126, 234, 0.4)";
              }}
            >
              Back to Training
            </button>
          </div>
        </div>
      ) : null}

      <main style={styles.testContent}>
        <div style={styles.testVideoWrap}>
          <video ref={setVideoEl} style={styles.testVideo} muted playsInline autoPlay />
          <div style={styles.cameraBadge}>{cameraReady ? "Camera live" : "Starting camera..."}</div>
        </div>

        <div style={styles.testResultBar}>
          <div>
            <div style={styles.resultLabel}>Predicted class</div>
            <div style={styles.resultValue}>{prediction}</div>
          </div>
          <div style={styles.confidenceBox}>
            <div style={styles.resultLabel}>Confidence</div>
            <div style={styles.confidenceValue}>{confidence}%</div>
            <div style={styles.confidenceTrack}>
              <div style={{ ...styles.confidenceFill, width: `${clamp01(confidence / 100) * 100}%` }} />
            </div>
          </div>
          <button
            onClick={() => navigate("/")}
            style={styles.ghostButton}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "#f1f5f9";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "transparent";
            }}
          >
            Back to Training
          </button>
        </div>

        <div style={styles.testMetaRow}>
          <div style={styles.metaCard}>
            <span style={{ color: "#64748b", fontSize: 13 }}>Status</span>
            <strong style={{ color: "#0f172a" }}>{status}</strong>
          </div>
          <div style={styles.metaCard}>
            <span style={{ color: "#64748b", fontSize: 13 }}>Classes</span>
            <strong style={{ color: "#0f172a" }}>{classes.length}</strong>
          </div>
          <div style={styles.metaCard}>
            <span style={{ color: "#64748b", fontSize: 13 }}>Samples</span>
            <strong style={{ color: "#0f172a" }}>{sampleCount}</strong>
          </div>
        </div>
      </main>
    </div>
  );
}

function AppShell() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<TrainingPage />} />
        <Route path="/test" element={<TestPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}

const styles = {
  pageShell: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "#0f172a",
    padding: 24,
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
  },

  testPageShell: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "#0f172a",
    padding: 24,
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
  },

  header: {
    background: "rgba(255, 255, 255, 0.95)",
    backdropFilter: "blur(10px)",
    borderRadius: 20,
    padding: "24px 32px",
    marginBottom: 24,
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.12)",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 16,
  },

  brand: {
    fontSize: 28,
    fontWeight: 700,
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    letterSpacing: "-0.5px",
  },

  subBrand: {
    fontSize: 14,
    color: "#64748b",
    marginTop: 4,
    fontWeight: 400,
  },

  headerPills: {
    display: "flex",
    gap: 12,
    flexWrap: "wrap",
    alignItems: "center",
  },

  pill: {
    padding: "8px 16px",
    borderRadius: 12,
    fontSize: 13,
    fontWeight: 500,
    transition: "all 0.2s ease",
  },

  pillSuccess: {
    background: "#d1fae5",
    color: "#065f46",
  },

  pillWarning: {
    background: "#fed7aa",
    color: "#9a3412",
  },

  pillMuted: {
    background: "#f1f5f9",
    color: "#475569",
  },

  grid3: {
    display: "grid",
    gridTemplateColumns: "1.2fr 1fr 0.9fr",
    gap: 24,
    height: "calc(100vh - 180px)",
  },

  panel: {
    background: "rgba(255, 255, 255, 0.98)",
    backdropFilter: "blur(20px)",
    borderRadius: 20,
    padding: 24,
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.1)",
    overflowY: "auto",
    transition: "transform 0.2s ease, box-shadow 0.2s ease",
  },

  panelHeaderRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
    paddingBottom: 16,
    borderBottom: "2px solid #f1f5f9",
  },

  panelTitle: {
    fontSize: 20,
    fontWeight: 700,
    color: "#0f172a",
    letterSpacing: "-0.3px",
  },

  subText: {
    fontSize: 14,
    color: "#64748b",
    marginBottom: 16,
    lineHeight: 1.6,
  },

  primaryButton: {
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "#fff",
    border: "none",
    padding: "12px 24px",
    borderRadius: 12,
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
    transition: "all 0.2s ease",
    boxShadow: "0 4px 14px rgba(102, 126, 234, 0.4)",
  },

  secondaryButton: {
    border: "2px solid #e2e8f0",
    background: "#ffffff",
    padding: "10px 18px",
    borderRadius: 12,
    fontSize: 14,
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.2s ease",
    color: "#475569",
  },

  dangerButton: {
    border: "2px solid #fecaca",
    background: "#fef2f2",
    padding: "10px 18px",
    borderRadius: 12,
    fontSize: 14,
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.2s ease",
    color: "#dc2626",
  },

  ghostButton: {
    border: "2px solid #e2e8f0",
    background: "transparent",
    padding: "10px 18px",
    borderRadius: 12,
    fontSize: 14,
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.2s ease",
    color: "#475569",
  },

  classList: {
    display: "grid",
    gap: 16,
  },

  classCard: {
    border: "2px solid #e5e7eb",
    borderRadius: 16,
    padding: 16,
    background: "linear-gradient(135deg, #fafafa 0%, #ffffff 100%)",
    transition: "all 0.2s ease",
  },

  classCardTop: {
    display: "flex",
    gap: 12,
    marginBottom: 12,
  },

  classInput: {
    flex: 1,
    border: "2px solid #e2e8f0",
    borderRadius: 12,
    padding: "10px 14px",
    fontSize: 14,
    fontWeight: 500,
    transition: "all 0.2s ease",
    outline: "none",
  },

  classMetaRow: {
    display: "flex",
    gap: 8,
    marginBottom: 12,
  },

  countChip: {
    background: "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)",
    padding: "6px 12px",
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 600,
    color: "#1e40af",
  },

  countChipMuted: {
    background: "#f1f5f9",
    padding: "6px 12px",
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 500,
    color: "#64748b",
  },

  collectRow: {
    display: "flex",
    gap: 12,
    marginBottom: 16,
  },

  fileButton: {
    border: "2px solid #e2e8f0",
    padding: "10px 18px",
    borderRadius: 12,
    fontSize: 14,
    fontWeight: 500,
    cursor: "pointer",
    background: "#ffffff",
    transition: "all 0.2s ease",
    color: "#475569",
    display: "inline-block",
  },

  trainVideoWrap: {
    position: "relative",
    borderRadius: 16,
    overflow: "hidden",
    marginBottom: 16,
    aspectRatio: "4/3",
    background: "#000",
    boxShadow: "0 10px 40px rgba(0, 0, 0, 0.2)",
  },

  trainVideo: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },

  cameraOverlay: {
    position: "absolute",
    bottom: 16,
    left: 16,
    background: "rgba(0, 0, 0, 0.75)",
    backdropFilter: "blur(10px)",
    color: "#fff",
    padding: "8px 16px",
    borderRadius: 10,
    fontSize: 13,
    fontWeight: 500,
  },

  cameraBadge: {
    position: "absolute",
    top: 16,
    right: 16,
    background: "rgba(16, 185, 129, 0.9)",
    backdropFilter: "blur(10px)",
    color: "#fff",
    padding: "8px 16px",
    borderRadius: 10,
    fontSize: 13,
    fontWeight: 600,
    boxShadow: "0 4px 14px rgba(16, 185, 129, 0.4)",
  },

  sampleScroller: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(100px, 1fr))",
    gap: 12,
    maxHeight: "280px",
    overflowY: "auto",
    padding: "8px 0",
  },

  sampleCard: {
    borderRadius: 12,
    overflow: "hidden",
    background: "#fff",
    border: "2px solid #e5e7eb",
    transition: "all 0.2s ease",
  },

  sampleImage: {
    width: "100%",
    aspectRatio: "1/1",
    objectFit: "cover",
  },

  sampleMetaRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 8,
  },

  sampleBadge: {
    fontSize: 11,
    background: "#dbeafe",
    color: "#1e40af",
    padding: "4px 8px",
    borderRadius: 6,
    fontWeight: 600,
  },

  sampleHint: {
    fontSize: 11,
    textAlign: "center",
    color: "#64748b",
    paddingBottom: 6,
    fontWeight: 500,
  },

  iconButton: {
    border: "none",
    background: "transparent",
    cursor: "pointer",
    color: "#ef4444",
    fontSize: 16,
    fontWeight: 700,
    transition: "all 0.2s ease",
    width: 24,
    height: 24,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },

  samplePlaceholder: {
    gridColumn: "1 / -1",
    textAlign: "center",
    color: "#94a3b8",
    padding: "32px 16px",
    fontSize: 14,
  },

  emptyState: {
    textAlign: "center",
    color: "#94a3b8",
    padding: "48px 24px",
    fontSize: 14,
    border: "2px dashed #e2e8f0",
    borderRadius: 16,
  },

  trainingCard: {
    border: "2px solid #e5e7eb",
    borderRadius: 16,
    padding: 20,
    background: "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
    marginTop: 16,
  },

  trainingStatusLabel: {
    fontSize: 13,
    color: "#64748b",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginBottom: 8,
  },

  trainingStatusText: {
    fontSize: 15,
    fontWeight: 600,
    marginBottom: 12,
    color: "#0f172a",
  },

  smallNote: {
    fontSize: 13,
    color: "#64748b",
    marginBottom: 16,
    lineHeight: 1.5,
  },

  trainActions: {
    display: "flex",
    gap: 12,
    marginTop: 16,
  },

  basicInfoBox: {
    background: "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)",
    borderRadius: 12,
    padding: 16,
    fontSize: 14,
    lineHeight: 1.6,
    color: "#475569",
    border: "2px solid #e2e8f0",
  },

  paramGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 16,
    marginBottom: 16,
  },

  paramBox: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    fontSize: 13,
    fontWeight: 600,
    color: "#475569",
  },

  numberInput: {
    border: "2px solid #e2e8f0",
    borderRadius: 10,
    padding: "8px 12px",
    fontSize: 14,
    fontWeight: 500,
    outline: "none",
    transition: "all 0.2s ease",
  },

  modeToggleWrap: {
    display: "flex",
    gap: 0,
    background: "#f1f5f9",
    borderRadius: 12,
    padding: 4,
  },

  toggleButton: {
    padding: "8px 16px",
    border: "none",
    borderRadius: 10,
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    transition: "all 0.2s ease",
    background: "transparent",
    color: "#64748b",
  },

  toggleActive: {
    background: "#ffffff",
    color: "#0f172a",
    boxShadow: "0 2px 8px rgba(0, 0, 0, 0.08)",
  },

  readyCard: {
    border: "2px solid #e5e7eb",
    borderRadius: 16,
    padding: 24,
    background: "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
    marginBottom: 20,
  },

  readyBadge: (isReady) => ({
    display: "inline-block",
    padding: "8px 16px",
    borderRadius: 999,
    fontSize: 13,
    fontWeight: 700,
    marginBottom: 16,
    background: isReady
      ? "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
      : "#f1f5f9",
    color: isReady ? "#065f46" : "#64748b",
  }),

  readyText: {
    fontSize: 14,
    color: "#475569",
    lineHeight: 1.6,
    marginBottom: 20,
  },

  readOnlyStats: {
    display: "grid",
    gap: 12,
    marginBottom: 20,
  },

  statRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "12px 16px",
    background: "#f8fafc",
    borderRadius: 12,
    fontSize: 14,
    border: "2px solid #f1f5f9",
  },

  linkRow: {
    textAlign: "center",
  },

  linkLike: {
    color: "#667eea",
    textDecoration: "none",
    fontSize: 14,
    fontWeight: 600,
    transition: "all 0.2s ease",
  },

  testContent: {
    display: "grid",
    gridTemplateColumns: "2fr 1fr",
    gap: 24,
  },

  testVideoWrap: {
    position: "relative",
    borderRadius: 20,
    overflow: "hidden",
    background: "#000",
    height: "calc(100vh - 280px)",
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.3)",
  },

  testVideo: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },

  testResultBar: {
    display: "flex",
    flexDirection: "column",
    gap: 20,
    background: "rgba(255, 255, 255, 0.98)",
    backdropFilter: "blur(20px)",
    padding: 24,
    borderRadius: 20,
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.1)",
  },

  resultLabel: {
    fontSize: 13,
    color: "#64748b",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginBottom: 8,
  },

  resultValue: {
    fontSize: 32,
    fontWeight: 700,
    color: "#0f172a",
    letterSpacing: "-0.5px",
  },

  confidenceBox: {
    padding: 20,
    background: "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)",
    borderRadius: 16,
    border: "2px solid #e2e8f0",
  },

  confidenceValue: {
    fontSize: 24,
    fontWeight: 700,
    color: "#667eea",
  },

  confidenceTrack: {
    height: 8,
    background: "#e2e8f0",
    borderRadius: 999,
    overflow: "hidden",
    marginTop: 12,
  },

  confidenceFill: {
    height: "100%",
    background: "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
    transition: "width 0.3s ease",
    borderRadius: 999,
  },

  testMetaRow: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 16,
    gridColumn: "1 / -1",
  },

  metaCard: {
    border: "2px solid #e5e7eb",
    padding: 16,
    borderRadius: 16,
    display: "flex",
    flexDirection: "column",
    gap: 8,
    fontSize: 14,
    background: "rgba(255, 255, 255, 0.98)",
    backdropFilter: "blur(20px)",
  },

  blockerOverlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0, 0, 0, 0.5)",
    backdropFilter: "blur(8px)",
    display: "grid",
    placeItems: "center",
    zIndex: 1000,
  },

  blockerCard: {
    background: "#fff",
    padding: 32,
    borderRadius: 20,
    width: 400,
    maxWidth: "90%",
    textAlign: "center",
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.3)",
  },
};
