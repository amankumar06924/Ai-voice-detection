
import React, { useState, useRef, useEffect } from 'react';
import './App.css';

const WEBSOCKET_URL = "ws://localhost:8000/ws/stream";

export default function ProxyPhoneVoiceDetector() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [callState, setCallState] = useState("IDLE"); // IDLE, RINGING, CONNECTED
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeChunkCount, setActiveChunkCount] = useState(0);

  const mediaRecorderRef = useRef(null);
  const socketRef = useRef(null);
  const chunkCounterRef = useRef(0);

  // WEBSOCKET HANDLERS
  const setupWebSocket = () => {
    return new Promise((resolve, reject) => {
      try {
        const socket = new WebSocket(WEBSOCKET_URL);
        socketRef.current = socket;

        socket.onopen = () => {
          console.log("Connected to Python backend WebSocket proxy");
          resolve(socket);
        };

        socket.onmessage = (event) => {
          const data = JSON.parse(event.data);
          setIsProcessing(false);
          setResult({
            verdict: data.verdict, // "AI_CLONE" or "HUMAN"
            ai_probability: data.ai_probability,
            human_probability: data.human_probability,
            timestamp: new Date().toLocaleTimeString()
          });
        };

        socket.onerror = (err) => {
          console.error("WebSocket Error:", err);
          setError("WebSocket connection failed. Ensure Python server is active.");
          reject(err);
        };

        socket.onclose = () => {
          console.log("WebSocket Disconnected");
        };
      } catch (err) {
        console.error("WS Setup error:", err);
        reject(err);
      }
    });
  };

  const closeWebSocket = () => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
  };

  // CALL CONTROLLERS
  const startCall = () => {
    setError(null);
    setResult(null);
    setCallState("RINGING");
  };

  const acceptCall = async () => {
    setCallState("CONNECTED");
    await setupWebSocket();
    start1SecContinuousStream();
  };

  const endCall = () => {
    stopContinuousStream();
    closeWebSocket();
    setCallState("IDLE");
    setIsProcessing(false);
    setActiveChunkCount(0);
    setResult(null);
  };

  // 1. FILE UPLOAD HANDLER (TRANSFERS THE WHOLE FILE VIA HTTP POST)
  const handleSendAudioFile = async () => {
    if (!selectedFile) {
      setError("Please select an audio file first.");
      return;
    }

    setError(null);
    setResult(null);
    setCallState("CONNECTED");
    setIsProcessing(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Send the whole file directly to your FastAPI backend server
      const response = await fetch("http://localhost:8000/api/detect-audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server status: ${response.status}`);
      }

      const data = await response.json();

      setIsProcessing(false);
      setResult({
        verdict: data.verdict, // "AI_CLONE" or "HUMAN"
        ai_probability: data.ai_probability,
        human_probability: data.human_probability,
        timestamp: new Date().toLocaleTimeString(),
      });
    } catch (err) {
      console.error("Audio Upload Error:", err);
      setIsProcessing(false);
      setError("Failed to upload audio file to server.");
    }
  };

  // 2. LIVE CALL ENGINE (STREAMS 16KHz AUDIO CHUNKS OVER WEBSOCKET)
  const start1SecContinuousStream = async () => {
    try {
      // Request audio stream forced to 16000Hz (16kHz) sample rate
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      chunkCounterRef.current = 0;

      // Use PCM/Audio WebM container at 16kHz
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000,
      });

      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0 && socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
          // Stream 1-second 16kHz chunk to backend over WebSocket
          socketRef.current.send(e.data);

          chunkCounterRef.current += 1;
          setActiveChunkCount(chunkCounterRef.current);

          // Trigger evaluation spinner state every 5 chunks
          if (chunkCounterRef.current % 5 === 0) {
            setIsProcessing(true);
          }
        }
      };

      // Emit chunk every 1000ms (1 second)
      mediaRecorder.start(1000);

    } catch (err) {
      console.error(err);
      setError("Microphone permission denied or 16kHz sample rate unsupported.");
      endCall();
    }
  };

  const stopContinuousStream = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      mediaRecorderRef.current.stop();
    }
  };

  useEffect(() => {
    return () => {
      stopContinuousStream();
      closeWebSocket();
    };
  }, []);


  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col items-center justify-center p-6 font-sans">
      <h1 className="text-xl md:text-2xl font-extrabold mb-6 text-slate-200">
        Real-Time In-Call AI Voice Detection
      </h1>

      <div className="w-full max-w-4xl rounded-3xl p-6 relative flex flex-col items-center">
        <div className="w-full rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-6 relative">
          
          {/* PHONE 1: CALLER */}
          <div>
            <div className="text-center mb-2">
              <h2 className="text-base font-bold text-slate-300">Caller (Device 1)</h2>
            </div>

            <div className="w-full md:w-72 h-120 bg-slate-900 rounded-[35px] p-4 border-4 border-slate-700 shadow-2xl flex flex-col justify-between">
              <div className="w-20 h-3 bg-slate-800 rounded-full mx-auto mb-2"></div>

              <div className="flex-1 flex flex-col justify-between py-2">
                
                {/* File Upload Option with Send Button */}
                <div className="bg-slate-800/60 p-2.5 rounded-xl border border-slate-700/50 space-y-2">
                  <label className="block text-[11px] font-medium text-slate-300">Use Audio File For Call</label>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setSelectedFile(e.target.files[0])}
                    className="block w-full text-[10px] text-slate-400 file:mr-2 file:py-1 file:px-2 file:rounded-md file:border-0 file:bg-blue-600 file:text-white cursor-pointer"
                  />
                  <button
                    onClick={handleSendAudioFile}
                    disabled={!selectedFile || callState === "CONNECTED"}
                    className="w-full py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 font-bold rounded-lg text-[11px] text-white transition flex items-center justify-center gap-1.5"
                  >
                    Send Audio File
                  </button>
                </div>

                {/* CALL INTERFACE SECTION */}
                <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800 flex flex-col items-center text-center space-y-3">
                  <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center text-2xl border border-slate-700">
                    OO
                  </div>
                  <div>
                    <h3 className="font-bold text-sm text-slate-100">Receiver</h3>
                    <p className="text-[10px] text-slate-400">
                      {callState === "IDLE" && "Ready to Call"}
                      {callState === "RINGING" && "Outgoing Call..."}
                      {callState === "CONNECTED" && " Call Active"}
                    </p>
                  </div>

                  {callState === "CONNECTED" && (
                    <div className="flex items-center gap-1 text-[10px] text-emerald-400 animate-pulse">
                      <span>Streaming (1s chunks)</span>
                      <span>• #{activeChunkCount}</span>
                    </div>
                  )}

                  {/* CALL BUTTON CONTROLS */}
                  <div className="w-full pt-2">
                    {callState === "IDLE" && (
                      <button
                        onClick={startCall}
                        className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 font-bold rounded-xl text-xs transition flex items-center justify-center gap-2"
                      >
                        Start Call
                      </button>
                    )}

                    {callState !== "IDLE" && (
                      <button
                        onClick={endCall}
                        className="w-full py-2 bg-red-600 hover:bg-red-500 font-bold rounded-xl text-xs transition flex items-center justify-center gap-2"
                      >
                        End Call
                      </button>
                    )}
                  </div>
                </div>

                
              </div>

              <div className="w-10 h-1 bg-slate-700 rounded-full mx-auto mt-1"></div>
            </div>
          </div>

          {/* PHONE 2: RECEIVER */}
          <div>
            <div className="text-center mb-2">
              <h2 className="text-base font-bold text-slate-300">Receiver (Device 2)</h2>
            </div>

            <div className="w-full md:w-72 h-120 bg-slate-900 rounded-[35px] p-4 border-4 border-slate-700 shadow-2xl flex flex-col justify-between">
              <div className="w-20 h-3 bg-slate-800 rounded-full mx-auto mb-2"></div>

              <div className="flex-1 flex flex-col justify-between py-2">
                
                {/* IDLE STATE */}
                {callState === "IDLE" && (
                  <div className="flex-1 flex flex-col justify-center items-center text-center text-slate-500 text-xs">
                    Phone Locked / Waiting for incoming call...
                  </div>
                )}

                {/* INCOMING RINGING STATE */}
                {callState === "RINGING" && (
                  <div className="flex-1 flex flex-col justify-center items-center text-center space-y-4">
                    <div className="w-16 h-16 bg-blue-600/20 text-blue-400 rounded-full flex items-center justify-center text-2xl animate-bounce">
                      📲
                    </div>
                    <div>
                      <h3 className="font-bold text-sm text-slate-100">Incoming Call</h3>
                      <p className="text-[10px] text-slate-400">Caller ID: Unknown</p>
                    </div>

                    <div className="w-full flex gap-2">
                      <button
                        onClick={acceptCall}
                        className="flex-1 py-2 bg-emerald-600 hover:bg-emerald-500 font-bold rounded-xl text-xs text-white transition animate-pulse"
                      >
                        Answer
                      </button>
                      <button
                        onClick={endCall}
                        className="flex-1 py-2 bg-red-600 hover:bg-red-500 font-bold rounded-xl text-xs text-white transition"
                      >
                        Decline
                      </button>
                    </div>
                  </div>
                )}

                {/* CONNECTED STATE & AI INSPECTION DISPLAY */}
                {callState === "CONNECTED" && (
                  <div className="flex-1 flex flex-col justify-between space-y-2">
                    
                    {/* Live Status Header */}
                    <div className="bg-slate-950 p-2 rounded-xl border border-slate-800 flex items-center justify-between px-3">
                      <span className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider">
                        ● Call Active
                      </span>
                      <button
                        onClick={endCall}
                        className="bg-red-600/80 hover:bg-red-600 text-white text-[10px] font-bold px-2.5 py-1 rounded-lg transition"
                      >
                        End Call
                      </button>
                    </div>

                    {/* AI Inspection Card */}
                    <div className="flex-1 bg-slate-950 rounded-2xl border border-slate-800 p-3 flex flex-col justify-center items-center text-center">
                      {isProcessing && (
                        <div className="space-y-1.5">
                          <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
                          <p className="text-[10px] text-blue-400">Evaluating 5-second Window...</p>
                        </div>
                      )}

                      {error && (
                        <p className="text-red-400 text-[10px] bg-red-950/50 p-2 rounded-lg">{error}</p>
                      )}

                      {!isProcessing && !result && !error && (
                        <p className="text-slate-500 text-[10px]">Listening to incoming audio stream...</p>
                      )}

                      {result && !isProcessing && (
                        <div className="w-full space-y-2">
                          <div className={`p-2 rounded-xl border ${
                            result.verdict === "AI_CLONE"
                              ? "bg-red-950/80 border-red-600 text-red-200 animate-pulse"
                              : "bg-emerald-950/80 border-emerald-600 text-emerald-200"
                          }`}>
                            <div className="text-base mb-0.5">
                              {result.verdict === "AI_CLONE" ? "🚨" : "🛡️"}
                            </div>
                            <h3 className="text-[10px] font-black uppercase">
                              {result.verdict === "AI_CLONE" ? "AI Impersonation Detected!" : "Safe Human Call"}
                            </h3>
                          </div>

                          <div className="text-left text-[10px] bg-slate-900 p-2 rounded-lg border border-slate-800 space-y-1">
                            <div className="flex justify-between">
                              <span className="text-slate-400">AI Confidence:</span>
                              <span className="font-bold text-red-400">{(result.ai_probability * 100).toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-slate-400">Human Confidence:</span>
                              <span className="font-bold text-emerald-400">{(result.human_probability * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Receiver Direct Hangup Button */}
                    <button
                      onClick={endCall}
                      className="w-full py-1.5 bg-red-600 hover:bg-red-500 font-bold rounded-xl text-xs text-white transition flex items-center justify-center gap-1.5"
                    >
                      🔴 Hang Up
                    </button>
                  </div>
                )}

              </div>

              <div className="w-10 h-1 bg-slate-700 rounded-full mx-auto mt-1"></div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
