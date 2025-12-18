import React from 'react';

const Navbar: React.FC = () => {
  const scrollToWaitlist = () => {
    document.getElementById('waitlist')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <nav className="w-full px-6 md:px-12 py-6 flex items-center justify-between z-50 text-sm bg-transparent">
      {/* Logo Section */}
      <div className="flex items-center gap-2">
        <div className="relative flex items-center justify-center w-8 h-8 bg-gradient-to-tr from-purple-600 to-blue-600 rounded-lg shadow-md shadow-purple-500/20">
           <span className="font-bold text-white text-lg font-serif">H</span>
        </div>
        <div className="flex flex-col leading-tight ml-2">
          <span className="font-bold text-gray-900 tracking-tight text-lg font-serif">Hemlo</span>
        </div>
      </div>

      {/* Center Links - Hidden on mobile, visible on desktop */}
      <div className="hidden md:flex items-center gap-8 text-gray-600 font-medium">
        <a href="#" className="hover:text-black transition-colors">Vision</a>
        <a href="#" className="hover:text-black transition-colors">Market</a>
        <a href="#" className="hover:text-black transition-colors">Team</a>
      </div>

      {/* Right Actions */}
      <div className="flex items-center gap-6">
        <button 
          onClick={scrollToWaitlist}
          className="border border-yellow-600 text-yellow-700 hover:bg-yellow-50 px-5 py-2 rounded-full font-medium transition-all text-xs md:text-sm"
        >
          Join Waitlist
        </button>
      </div>
    </nav>
  );
};

export default Navbar;