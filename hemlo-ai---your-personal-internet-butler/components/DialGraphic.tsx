import React from 'react';

const DialGraphic: React.FC = () => {
  return (
    <div className="relative w-[300px] h-[300px] md:w-[500px] md:h-[500px] lg:w-[600px] lg:h-[600px] select-none pointer-events-none">
      
      {/* Container to position the partial circle on the right/bottom */}
      <div className="absolute inset-0 flex items-center justify-center">
        
        {/* The Graphic SVG */}
        <svg 
          viewBox="0 0 600 600" 
          className="w-full h-full transform translate-x-1/4 translate-y-1/6"
        >
          <defs>
            <linearGradient id="dialGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#e0e7ff" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#c7d2fe" stopOpacity="0.1" />
            </linearGradient>
            
            <linearGradient id="innerCircleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#a5b4fc" stopOpacity="0.2" />
              <stop offset="100%" stopColor="#818cf8" stopOpacity="0.6" />
            </linearGradient>

             <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="10" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Outermost dashed track - very faint */}
          <circle 
            cx="300" 
            cy="300" 
            r="280" 
            fill="none" 
            stroke="#e0e7ff" 
            strokeWidth="1" 
            strokeOpacity="0.2"
            strokeDasharray="4 4"
          />

          {/* The Tick Marks Ring */}
          {/* We'll generate tick marks programmatically */}
          <g transform="translate(300, 300)">
             {Array.from({ length: 60 }).map((_, i) => (
               <line
                 key={i}
                 x1="0"
                 y1="-220"
                 x2="0"
                 y2="-240"
                 stroke={i % 5 === 0 ? "rgba(255,255,255,0.8)" : "rgba(255,255,255,0.3)"}
                 strokeWidth={i % 5 === 0 ? "2" : "1"}
                 transform={`rotate(${i * 6})`}
               />
             ))}
          </g>

          {/* Main Solid Arc Background */}
          <circle 
            cx="300" 
            cy="300" 
            r="200" 
            fill="url(#dialGradient)"
            className="drop-shadow-2xl"
          />

          {/* Inner Decorative Rings */}
          <circle 
            cx="300" 
            cy="300" 
            r="160" 
            fill="none" 
            stroke="white" 
            strokeWidth="1" 
            strokeOpacity="0.3" 
          />
          
          <circle 
            cx="300" 
            cy="300" 
            r="130" 
            fill="url(#innerCircleGradient)"
            style={{ mixBlendMode: 'overlay' }}
          />
          
          {/* The "active" segment simulation */}
          <path
            d="M 300 100 A 200 200 0 0 1 500 300"
            fill="none"
            stroke="white"
            strokeWidth="2"
            strokeOpacity="0.5"
            strokeLinecap="round"
          />
          
          {/* Subtle inner lines */}
           <g transform="translate(300, 300)">
             {Array.from({ length: 30 }).map((_, i) => (
               <line
                 key={`inner-${i}`}
                 x1="0"
                 y1="-130"
                 x2="0"
                 y2="-150"
                 stroke="rgba(255,255,255,0.15)"
                 strokeWidth="1"
                 transform={`rotate(${i * 6 - 45})`}
               />
             ))}
          </g>

        </svg>

        {/* 3D-ish Overlay effect using CSS for better blur handling than SVG filters sometimes */}
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-white/10 to-transparent pointer-events-none mix-blend-overlay"></div>
      </div>
    </div>
  );
};

export default DialGraphic;