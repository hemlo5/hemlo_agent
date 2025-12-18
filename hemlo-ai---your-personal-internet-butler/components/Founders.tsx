import React from 'react';
import { Quote } from 'lucide-react';

const Founders: React.FC = () => {
  return (
    <section className="w-full py-24 bg-gradient-to-br from-purple-700 via-indigo-800 to-purple-900 text-white">
      <div className="max-w-7xl mx-auto px-6">
        
        <div className="text-center mb-16">
          <h2 className="font-serif text-4xl md:text-[40px] font-bold mb-4">Built by Teens, For the Future</h2>
          <p className="text-purple-200 text-lg max-w-2xl mx-auto font-light">
            Two 17-year-olds from India, coding full-time. No fancy degrees, just pure grit and vision to make AI your daily butler.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          
          {/* Founder 1 */}
          <div className="group relative bg-white/10 backdrop-blur-md border border-white/20 rounded-3xl p-8 hover:bg-white/15 transition-all duration-300 hover:-translate-y-2">
            <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/30 rounded-full blur-2xl -mr-10 -mt-10 pointer-events-none group-hover:bg-purple-500/50 transition-colors"></div>
            
            <div className="flex items-center gap-6 mb-6 relative z-10">
              <div className="w-20 h-20 rounded-full bg-gray-200 overflow-hidden border-2 border-purple-400/50 flex items-center justify-center bg-cover bg-center" style={{backgroundImage: 'url("https://images.unsplash.com/photo-1595152772835-219674b2a8a6?auto=format&fit=crop&q=80&w=200&h=200")'}}>
                {/* Fallback if image fails or for semantics */}
              </div>
              <div>
                <h3 className="text-2xl font-bold">Aniket</h3>
                <p className="text-purple-300 font-medium">CEO & Visionary</p>
                <div className="inline-block px-2 py-0.5 mt-2 rounded bg-purple-500/20 border border-purple-500/30 text-[10px] uppercase tracking-wider text-purple-200">
                  17 • Commerce Stream
                </div>
              </div>
            </div>
            
            <div className="relative z-10">
              <Quote size={24} className="text-purple-400/40 absolute -top-3 -left-2 rotate-180" />
              <p className="text-gray-200 italic pl-6 leading-relaxed">
                "The visionary hustler. Turned frustration with slow AIs into Hemlo's ecosystem dream. Handles growth, partnerships, and making it user-simple."
              </p>
            </div>
          </div>

          {/* Founder 2 */}
          <div className="group relative bg-white/10 backdrop-blur-md border border-white/20 rounded-3xl p-8 hover:bg-white/15 transition-all duration-300 hover:-translate-y-2">
            <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/30 rounded-full blur-2xl -mr-10 -mt-10 pointer-events-none group-hover:bg-indigo-500/50 transition-colors"></div>
            
            <div className="flex items-center gap-6 mb-6 relative z-10">
              <div className="w-20 h-20 rounded-full bg-gray-200 overflow-hidden border-2 border-indigo-400/50 flex items-center justify-center bg-cover bg-center" style={{backgroundImage: 'url("https://images.unsplash.com/photo-1544717305-2782549b5136?auto=format&fit=crop&q=80&w=200&h=200")'}}>
              </div>
              <div>
                <h3 className="text-2xl font-bold">Amit</h3>
                <p className="text-indigo-300 font-medium">CTO & Code Wizard</p>
                <div className="inline-block px-2 py-0.5 mt-2 rounded bg-indigo-500/20 border border-indigo-500/30 text-[10px] uppercase tracking-wider text-indigo-200">
                  17 • Tech Stream
                </div>
              </div>
            </div>
            
            <div className="relative z-10">
              <Quote size={24} className="text-indigo-400/40 absolute -top-3 -left-2 rotate-180" />
              <p className="text-gray-200 italic pl-6 leading-relaxed">
                "The code wizard. Built the DOM magic from scratch – Playwright + Llama for bulletproof automation. Loves solving 'impossible' web tasks."
              </p>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

export default Founders;