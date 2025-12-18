import React from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Vision from './components/Vision';
import Business from './components/Business';
import Founders from './components/Founders';
import Footer from './components/Footer';

const App: React.FC = () => {
  return (
    <div className="relative min-h-screen w-full bg-white selection:bg-purple-200">
      {/* Main Container */}
      <div className="relative w-full max-w-[1600px] mx-auto bg-white min-h-screen flex flex-col overflow-hidden shadow-2xl">
        
        {/* Background Gradients for Hero Area */}
        <div className="absolute top-0 left-0 w-full h-[120vh] z-0 pointer-events-none overflow-hidden">
           <div className="absolute -left-[10%] bottom-0 w-[60%] h-[80%] bg-[#c7d2fe] opacity-60 blur-[120px] rounded-full mix-blend-multiply"></div>
           <div className="absolute left-[20%] top-[20%] w-[40%] h-[60%] bg-[#e0e7ff] opacity-60 blur-[100px] rounded-full"></div>
           <div className="absolute right-0 top-0 w-[60%] h-[80%] bg-white opacity-80 blur-[80px] rounded-full"></div>
        </div>

        {/* Content */}
        <div className="relative z-10 flex flex-col h-full">
          <Navbar />
          <Hero />
          <Vision />
          <Business />
          <Founders />
          <Footer />
        </div>
      </div>
    </div>
  );
};

export default App;