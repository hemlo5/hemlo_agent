import React, { useState } from 'react';
import { WebGLShader } from "./components/ui/web-gl-shader";
import { LiquidButton } from './components/ui/liquid-glass-button';

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [recentHemlos, setRecentHemlos] = useState<string[]>([]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    
    // Add new prompt to the beginning of the list
    setRecentHemlos((prev) => [prompt, ...prev]);
    console.log("Prompt submitted:", prompt);
    setPrompt("");
  };

  return (
    <div className="relative flex w-full min-h-screen flex-col items-center justify-center bg-black text-foreground">
      {/* Background Shader */}
      <WebGLShader/> 
      
      {/* Scrollable Content Container */}
      <div className="relative z-10 w-full max-w-3xl px-4 py-10 max-h-screen overflow-y-auto scrollbar-hide">
        {/* Simplified container without borders/bg */}
        <div className="w-full mx-auto">
          <main className="relative py-12 px-6 md:px-12 flex flex-col items-center gap-10">
            
            {/* Bold White Title */}
            <h1 className="text-white text-center text-7xl md:text-[clamp(4rem,10vw,8rem)] font-black tracking-tighter drop-shadow-[0_0_30px_rgba(255,255,255,0.4)]">
              HEMLO
            </h1>
            
            <form onSubmit={handleSubmit} className="w-full max-w-xl flex flex-col gap-6 z-20">
              <div className="relative group">
                {/* Glow effect behind textarea */}
                <div className="absolute -inset-0.5 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-lg blur opacity-20 group-hover:opacity-60 transition duration-1000 group-hover:duration-200"></div>
                
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Ask me to perform your daily boring task"
                  className="relative w-full h-32 bg-[#09090b]/90 text-white placeholder-zinc-500 border border-zinc-800 rounded-lg p-5 focus:outline-none focus:ring-1 focus:ring-white/20 focus:border-white/20 resize-none text-lg shadow-inner transition-all"
                />
              </div>
              
              <div className="flex justify-center"> 
                <LiquidButton 
                  type="submit"
                  className="text-white border border-white/20 rounded-full px-12 hover:border-white/40" 
                  size={'xl'}
                >
                  Go
                </LiquidButton> 
              </div> 
            </form>

            {/* Recent Hemlos List */}
            {recentHemlos.length > 0 && (
              <div className="w-full max-w-xl z-20 mt-2 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <h2 className="text-zinc-500 text-xs font-bold uppercase tracking-widest mb-4 text-center">
                  Recent Hemlos
                </h2>
                <div className="flex flex-col gap-3">
                  {recentHemlos.map((hemla, index) => (
                    <div 
                      key={index} 
                      className="group relative overflow-hidden rounded-lg border border-white/10 bg-white/5 p-4 transition-all hover:bg-white/10 hover:border-white/20"
                    >
                      <p className="text-sm text-zinc-300 group-hover:text-white transition-colors break-words">
                        {hemla}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </main>
        </div>
      </div>
    </div>
  )
}