import React from 'react';
import { Play, Sparkles } from 'lucide-react';
import DialGraphic from './DialGraphic';

const Hero: React.FC = () => {
  return (
    <main className="flex-1 flex flex-col items-center justify-center w-full max-w-7xl mx-auto px-6 lg:px-12 py-12 lg:py-0 min-h-[80vh] relative">
      
      {/* Centered Graphic Background */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -z-10 opacity-40 scale-125 lg:scale-150 pointer-events-none">
         <DialGraphic />
      </div>

      {/* Content */}
      <div className="flex flex-col justify-center items-center z-10 w-full text-center max-w-4xl">
        
        {/* Badge */}
        <div className="inline-flex items-center gap-2 bg-white/40 backdrop-blur-md border border-gray-200 text-gray-800 px-3 py-1.5 rounded-full text-xs font-medium mb-8 shadow-sm ring-1 ring-black/5">
          <Sparkles size={12} className="text-yellow-500 fill-yellow-500" />
          <span className="tracking-wide">Beta Version is Live!</span>
        </div>

        {/* Heading */}
        <div className="mb-6">
          <h1 className="text-5xl md:text-7xl lg:text-[6rem] leading-[1.1] font-serif text-black tracking-tight drop-shadow-sm">
            <span className="block italic font-light">You're personal butler</span>
            <span className="block italic font-light text-gray-800">for everything</span>
          </h1>
        </div>

        {/* Subheading */}
        <p className="text-gray-600 text-sm md:text-base leading-relaxed max-w-xl mb-10 font-light mx-auto">
          Say goodbye to repetitive tasks. Our AI-driven platform streamlines your workflows so your team can focus on what really matters.
        </p>

        {/* Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-4">
          <button className="bg-black text-white hover:bg-gray-800 px-8 py-3 rounded-full font-semibold transition-all shadow-lg hover:shadow-xl text-sm md:text-base">
            See It in Action
          </button>
          
          <button className="flex items-center gap-2 bg-white/60 backdrop-blur-sm border border-gray-200 text-gray-900 hover:bg-white px-8 py-3 rounded-full font-medium transition-all text-sm md:text-base group shadow-sm">
            <span>Demo</span>
            <Play size={14} className="fill-gray-900 group-hover:scale-110 transition-transform" />
          </button>
        </div>
      </div>

    </main>
  );
};

export default Hero;