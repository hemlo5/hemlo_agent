import React from 'react';
import { Send, Twitter, Linkedin } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="w-full bg-[#05060e] py-16 px-6 text-white relative overflow-hidden" id="waitlist">
      {/* Background glow effect */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[800px] h-[300px] bg-purple-900/20 blur-[120px] rounded-full pointer-events-none"></div>

      <div className="max-w-4xl mx-auto text-center relative z-10">
        
        <h2 className="font-serif text-3xl md:text-4xl font-bold mb-4">Ready to Automate Your Life?</h2>
        <p className="text-gray-400 mb-10 max-w-xl mx-auto">
          Join the waitlist today. Get early access and lifetime free membership for the beta.
        </p>

        {/* Waitlist Form */}
        <div className="max-w-md mx-auto mb-16">
          <form className="flex flex-col sm:flex-row gap-3" onSubmit={(e) => e.preventDefault()}>
            <input 
              type="email" 
              placeholder="Enter your email address" 
              className="flex-1 bg-white/10 border border-white/10 rounded-full px-6 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 transition-colors"
            />
            <button className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-medium px-6 py-3 rounded-full transition-all shadow-lg shadow-purple-900/30 flex items-center justify-center gap-2 group">
              <span>Get Access</span>
              <Send size={16} className="group-hover:translate-x-1 transition-transform" />
            </button>
          </form>
          <p className="text-xs text-gray-500 mt-4">
            No spam, ever. Unsubscribe anytime.
          </p>
        </div>

        {/* Divider */}
        <div className="h-px w-full bg-white/10 mb-8"></div>

        {/* Bottom Bar */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="text-sm text-gray-500">
            &copy; 2025 Hemlo AI – Your Digital Butler.
          </div>
          
          <div className="flex items-center gap-6">
             <a href="#" className="text-gray-500 hover:text-white transition-colors p-2 rounded-full hover:bg-white/5">
               <Twitter size={20} />
             </a>
             <a href="#" className="text-gray-500 hover:text-white transition-colors p-2 rounded-full hover:bg-white/5">
               <Linkedin size={20} />
             </a>
          </div>
        </div>

      </div>
    </footer>
  );
};

export default Footer;