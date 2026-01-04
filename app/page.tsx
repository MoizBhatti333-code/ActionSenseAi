"use client";

import React, { useState, ChangeEvent } from 'react';
import { Upload, Loader2, Play, Activity, Type, RefreshCw, Sparkles, Image as ImageIcon } from 'lucide-react';

// --- Types ---
interface Annotation {
  time: string;
  text: string;
}

interface AnalysisResult {
  action_class: string;
  confidence: string;
  annotations: Annotation[];
}

export default function Home() {
  // --- State ---
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  // --- Handlers ---
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);

    try {
      // Create form data for API request
      const formData = new FormData();
      formData.append('image', file);

      // Call the backend API
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Analysis failed');
      }

      const data: AnalysisResult = await response.json();
      setResult(data);

    } catch (error) {
      console.error('Analysis error:', error);
      alert(error instanceof Error ? error.message : "Error during analysis");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen text-slate-900 font-sans selection:bg-indigo-100 selection:text-indigo-700">
      
      {/* Header with Backdrop Blur */}
      <header className="sticky top-0 z-20 bg-white/70 backdrop-blur-xl border-b border-slate-200/60 shadow-sm supports-[backdrop-filter]:bg-white/60">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center gap-3">
          <div className="p-2 bg-gradient-to-tr from-indigo-600 to-blue-500 rounded-xl text-white shadow-lg shadow-indigo-200">
            <Activity size={22} className="animate-pulse-slow"/>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-800">
            Action<span className="bg-gradient-to-r from-indigo-600 to-blue-500 bg-clip-text text-transparent">Sense</span> AI
          </h1>
          <div className="ml-auto hidden sm:flex items-center gap-2 text-sm font-medium text-slate-600 bg-slate-100/80 border border-slate-200 px-4 py-1.5 rounded-full">
            <Sparkles size={14} className="text-indigo-500"/>
             CNN + LSTM Model
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-10">
        <div className="grid lg:grid-cols-12 gap-8 h-auto lg:h-[calc(100vh-160px)] min-h-[600px]">
          
          {/* --- LEFT COLUMN (Inputs & Video) --- span-7 = approx 58% width */}
          <div className="lg:col-span-7 flex flex-col gap-6 h-full">
            
            {/* Enhanced Upload Area */}
            {!preview && (
              <div className="group flex-1 flex flex-col items-center justify-center border-[3px] border-dashed border-slate-300/80 hover:border-indigo-400/80 rounded-3xl bg-white/50 hover:bg-indigo-50/30 transition-all duration-500 ease-out relative overflow-hidden">
                <input 
                  type="file" 
                  accept="image/*"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                />
                
                {/* Decorative background blurring blobs */}
                <div className="absolute -top-10 -left-10 w-40 h-40 bg-indigo-100/50 rounded-full blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>
                <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-blue-100/50 rounded-full blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-700 delay-100"></div>

                <div className="relative z-20 flex flex-col items-center transition-transform duration-300 group-hover:scale-105">
                  <div className="w-24 h-24 bg-gradient-to-tr from-indigo-100 to-white shadow-inner border border-indigo-50 text-indigo-500 rounded-2xl flex items-center justify-center mb-6 group-hover:shadow-indigo-200/50 group-hover:text-indigo-600 transition-all">
                    <Upload size={40} className="group-hover:animate-bounce-slow" />
                  </div>
                  <h3 className="text-2xl font-bold text-slate-800 mb-3">Upload Image Analysis</h3>
                  <p className="text-slate-500 max-w-sm text-center leading-relaxed">
                    Drop your <span className="font-semibold text-indigo-600">JPEG</span>, <span className="font-semibold text-indigo-600">PNG</span>, or <span className="font-semibold text-indigo-600">WebP</span> file here to begin the CNN recognition process.
                  </p>
                </div>
              </div>
            )}

            {/* Enhanced Image Display Container */}
            {preview && (
              <div className="relative bg-slate-50 rounded-3xl overflow-hidden shadow-2xl shadow-slate-900/10 flex-1 flex flex-col justify-center group ring-1 ring-slate-900/5">
                <img 
                  src={preview} 
                  alt="Uploaded preview"
                  className="w-full h-full max-h-[600px] object-contain" 
                />
                {/* Glassmorphic Reset Button */}
                <button 
                    onClick={handleReset}
                    className="absolute top-5 right-5 z-10 flex items-center gap-2 bg-white/90 backdrop-blur-md border border-slate-200 text-slate-700 px-4 py-2 rounded-full hover:bg-white hover:shadow-lg transition-all opacity-0 group-hover:opacity-100 translate-y-2 group-hover:translate-y-0 duration-300"
                >
                    <RefreshCw size={16} />
                    <span className="text-sm font-medium">New Image</span>
                </button>
              </div>
            )}

            {/* Enhanced Analyze Button */}
            {file && !result && (
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="relative overflow-hidden w-full bg-gradient-to-r from-indigo-600 via-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white py-5 rounded-2xl font-bold text-xl shadow-xl shadow-indigo-200/60 hover:shadow-indigo-300/70 transition-all active:scale-[0.99] flex items-center justify-center gap-3 disabled:opacity-80 disabled:cursor-not-allowed group"
              >
                <div className="absolute inset-0 bg-white/20 skew-x-12 -translate-x-full group-hover:translate-x-[200%] transition-transform duration-1000 ease-in-out"></div>
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={24} />
                    <span>Processing Neural Network...</span>
                  </>
                ) : (
                  <>
                    <Play fill="currentColor" size={24} className="group-hover:translate-x-1 transition-transform"/>
                    <span>Run Model Analysis</span>
                  </>
                )}
              </button>
            )}
          </div>

          {/* --- RIGHT COLUMN (Results Panel) --- span-5 = approx 42% width */}
          <div className="lg:col-span-5 bg-white/80 backdrop-blur-lg rounded-3xl border border-white/40 shadow-xl shadow-slate-200/50 flex flex-col overflow-hidden h-[600px] lg:h-full ring-1 ring-slate-100">
            
            {/* Panel Header */}
            <div className="px-6 py-5 border-b border-slate-100 bg-gradient-to-r from-slate-50/80 to-white/80">
              <h2 className="text-lg font-bold text-slate-800 flex items-center gap-3">
                <div className="p-2 bg-indigo-100 text-indigo-600 rounded-lg">
                 <ImageIcon size={20} />
                </div>
                Intelligence Output
              </h2>
            </div>

            <div className="flex-1 overflow-y-auto p-0 scrollbar-thin scrollbar-thumb-indigo-100 scrollbar-track-transparent hover:scrollbar-thumb-indigo-200">
              
              {/* Enhanced Loading State */}
              {loading && (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-6 p-8 text-center">
                  <div className="relative">
                    <div className="absolute inset-0 bg-indigo-100 rounded-full blur-xl animate-pulse"></div>
                    <Loader2 size={50} className="animate-spin text-indigo-600 relative z-10" />
                  </div>
                  <div className="space-y-2 animate-pulse">
                    <p className="text-lg font-semibold text-slate-800">Analyzing Image</p>
                    <p className="text-sm">Extracting features through CNN layers...</p>
                  </div>
                </div>
              )}

              {/* Enhanced Empty State */}
              {!loading && !result && (
                <div className="h-full flex flex-col items-center justify-center text-slate-400 p-10 text-center gap-4">
                  <div className="w-20 h-20 bg-gradient-to-tr from-slate-100 to-white border border-slate-50 shadow-sm rounded-[2rem] flex items-center justify-center">
                    <ImageIcon size={32} className="text-slate-300" />
                  </div>
                  <p className="text-lg font-medium text-slate-500">Results pending analysis</p>
                </div>
              )}

              {/* Results Display */}
              {result && (
                <div className="divide-y divide-slate-100 animate-in fade-in slide-in-from-bottom-6 duration-700">
                  
                  {/* 1. Main Action Class (Hero Section) */}
                  <div className="p-8 bg-gradient-to-br from-indigo-50/50 via-white to-blue-50/30 relative overflow-hidden">
                     {/* Subtle background pattern */}
                    <Activity className="absolute -right-6 -bottom-6 text-indigo-100/80 opacity-50 w-32 h-32" />
                    
                    <div className="relative z-10">
                        <span className="inline-flex items-center gap-1.5 text-xs font-bold text-indigo-600 uppercase tracking-wider bg-indigo-100/80 px-3 py-1 rounded-full mb-3">
                            <Sparkles size={12} /> Detected Action
                        </span>
                        <div className="text-4xl font-extrabold text-slate-900 leading-tight tracking-tight">
                        {result.action_class}
                        </div>
                        
                        {/* Enhanced Confidence Meter */}
                        <div className="mt-6">
                            <div className="flex justify-between text-sm font-medium mb-2">
                                <span className="text-slate-500">Model Confidence</span>
                                <span className="text-indigo-700">{result.confidence}</span>
                            </div>
                            <div className="h-3 w-full bg-slate-200/70 rounded-full overflow-hidden p-0.5 box-content border border-slate-200/50">
                                <div 
                                    className="h-full bg-gradient-to-r from-indigo-500 to-blue-500 rounded-full shadow-sm transition-all duration-1000 ease-out relative" 
                                    style={{ width: result.confidence }}
                                >
                                    <div className="absolute inset-0 bg-white/30 animate-shimmer" style={{backgroundSize: '200% 100%'}}></div>
                                </div>
                            </div>
                        </div>
                    </div>
                  </div>

                  {/* 2. Annotations List */}
                  <div className="p-6 bg-white/60">
                    <h3 className="text-sm font-bold text-slate-700 uppercase tracking-wider mb-6 flex items-center gap-2 pl-2">
                      <Type size={16} className="text-indigo-500" />
                      Detected Regions
                    </h3>
                    
                    <div className="space-y-3">
                      {result.annotations.map((item, idx) => (
                        <div 
                          key={idx}
                          className="w-full text-left flex items-start gap-4 bg-white border border-slate-100 p-4 rounded-2xl shadow-sm"
                        >
                          {/* Region Label */}
                          <div className="flex-shrink-0 font-mono text-xs font-bold text-indigo-600 bg-indigo-50 border border-indigo-100 px-3 py-1.5 rounded-lg">
                            {item.time}
                          </div>
                          {/* Description Text */}
                          <p className="text-[15px] text-slate-600 leading-relaxed font-medium pt-0.5">
                            {item.text}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                </div>
              )}
            </div>
          </div>

        </div>
      </main>

      {/* Add subtle tailwind animation for shimmer effect */}
      <style jsx global>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
        .animate-pulse-slow {
          animation: pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
         @keyframes bounce-slow {
            0%, 100% { transform: translateY(-5%); animation-timing-function: cubic-bezier(0.8, 0, 1, 1); }
            50% { transform: translateY(0); animation-timing-function: cubic-bezier(0, 0, 0.2, 1); }
        }
        .animate-bounce-slow {
            animation: bounce-slow 2s infinite;
        }
      `}</style>
    </div>
  );
}