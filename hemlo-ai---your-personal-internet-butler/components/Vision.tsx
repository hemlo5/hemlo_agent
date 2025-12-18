import React from 'react';
import { 
  Plane, Download, Mail, Check, MousePointer, 
  Cpu, Search, Calendar, FileText, ShoppingCart, 
  CreditCard, Loader, Send
} from 'lucide-react';

const Vision: React.FC = () => {
  return (
    <section className="relative w-full py-24 bg-blue-50 overflow-hidden">
        {/* Background Gradients */}
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none opacity-40">
            <div className="absolute -top-[20%] -left-[10%] w-[120%] h-[60%] bg-blue-100 rounded-[50%] blur-[80px]"></div>
            <div className="absolute top-[40%] -right-[10%] w-[80%] h-[80%] bg-indigo-100 rounded-[50%] blur-[100px]"></div>
        </div>

        <div className="max-w-7xl mx-auto px-6 relative z-10">
            
            <div className="text-center mb-24">
                <h2 className="font-serif text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                    Our Vision: Your Digital Life, Automated
                </h2>
            </div>

            {/* Feature 1: Prompt -> Action (Browser Automation) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-32">
                {/* Visual */}
                <div className="order-2 lg:order-1 relative h-[400px] w-full bg-white rounded-3xl shadow-xl border border-blue-100 overflow-hidden flex flex-col group">
                    {/* Scene Container */}
                    <div className="absolute inset-0 bg-gray-50 flex flex-col items-center justify-center p-6">
                        
                        {/* 1. The Prompt Interface (Fades out or moves up) */}
                        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 w-[90%] max-w-sm z-20 animate-prompt-sequence">
                            <div className="bg-white/90 backdrop-blur border border-gray-200 p-3 rounded-2xl shadow-lg flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white shrink-0">
                                    <span className="font-serif font-bold text-xs">H</span>
                                </div>
                                <div className="flex-1 text-sm text-gray-800 font-medium overflow-hidden whitespace-nowrap">
                                    <span className="animate-typing-text inline-block overflow-hidden align-bottom">book flight to NYC for Friday...</span>
                                </div>
                                <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-blue-600 shrink-0">
                                    <Send size={14} />
                                </div>
                            </div>
                        </div>

                        {/* 2. The Browser Window (Slides up) */}
                        <div className="w-full max-w-md bg-white rounded-xl shadow-2xl border border-gray-200 overflow-hidden animate-browser-sequence opacity-0 transform translate-y-10">
                            {/* Browser Header */}
                            <div className="bg-gray-100 px-4 py-2 flex items-center gap-2 border-b border-gray-200">
                                <div className="flex gap-1.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-red-400"></div>
                                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-400"></div>
                                    <div className="w-2.5 h-2.5 rounded-full bg-green-400"></div>
                                </div>
                                <div className="flex-1 mx-4 bg-white rounded h-5 text-[10px] flex items-center px-2 text-gray-400 shadow-sm">
                                    travel-site.com/search
                                </div>
                            </div>
                            
                            {/* Browser Body - Flight Form */}
                            <div className="p-6 space-y-4">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="h-4 w-24 bg-blue-100 rounded"></div>
                                    <div className="h-8 w-8 bg-blue-50 rounded-full"></div>
                                </div>
                                
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-1">
                                        <div className="text-[10px] text-gray-400 uppercase font-bold">From</div>
                                        <div className="h-10 bg-gray-50 border border-gray-200 rounded px-3 flex items-center text-sm font-medium text-gray-800 relative overflow-hidden">
                                            <span className="animate-fill-from opacity-0">Mumbai (BOM)</span>
                                        </div>
                                    </div>
                                    <div className="space-y-1">
                                        <div className="text-[10px] text-gray-400 uppercase font-bold">To</div>
                                        <div className="h-10 bg-gray-50 border border-gray-200 rounded px-3 flex items-center text-sm font-medium text-gray-800 relative overflow-hidden">
                                            <span className="animate-fill-to opacity-0">New York (JFK)</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-2">
                                    <button className="w-full bg-blue-600 text-white h-10 rounded font-medium text-sm shadow-md relative overflow-hidden group-btn">
                                        <span className="relative z-10">Search Flights</span>
                                        <div className="absolute inset-0 bg-blue-700 transform scale-x-0 origin-left group-btn-active:scale-x-100 transition-transform"></div>
                                    </button>
                                </div>
                            </div>

                            {/* Results Overlay (Appears after search) */}
                            <div className="absolute inset-0 top-9 bg-white z-10 animate-show-results opacity-0 flex flex-col">
                                <div className="p-4 border-b border-gray-100 flex items-center justify-between bg-blue-50/50">
                                    <span className="text-xs font-bold text-blue-800">Results: 14 flights</span>
                                </div>
                                <div className="p-4 space-y-3">
                                    <div className="p-3 border border-green-200 bg-green-50 rounded-lg flex justify-between items-center shadow-sm relative overflow-hidden">
                                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-green-500"></div>
                                        <div>
                                            <div className="text-xs font-bold text-gray-800">Emirates</div>
                                            <div className="text-[10px] text-gray-500">10:00 AM - 4:00 PM</div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-sm font-bold text-green-700">$850</div>
                                            <div className="text-[10px] bg-green-200 text-green-800 px-1.5 py-0.5 rounded inline-block mt-1">Booked</div>
                                        </div>
                                    </div>
                                    <div className="p-3 border border-gray-100 rounded-lg flex justify-between items-center opacity-50">
                                        <div>
                                            <div className="text-xs font-bold text-gray-800">British Airways</div>
                                        </div>
                                        <div className="text-sm font-bold text-gray-400">$920</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Cursor */}
                        <div className="absolute z-50 animate-cursor-flow pointer-events-none opacity-0">
                            <MousePointer size={24} className="fill-black text-white drop-shadow-md" />
                        </div>

                    </div>
                </div>

                {/* Text */}
                <div className="order-1 lg:order-2 lg:pl-8">
                    <h3 className="font-serif text-3xl md:text-4xl font-bold mb-6 text-gray-900">Just Prompt It</h3>
                    <p className="text-lg text-gray-600 leading-relaxed font-light">
                        Imagine saying <span className="font-medium text-blue-700 bg-blue-50 px-1 rounded">"book flight to NYC"</span> and Hemlo opens the browser, navigates, searches, and books. Real automation, not just text generation.
                    </p>
                </div>
            </div>

            {/* Feature 2: Full Ecosystem (Grid of possibilities) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-32">
                {/* Text */}
                <div className="order-1 lg:pr-8">
                    <h3 className="font-serif text-3xl md:text-4xl font-bold mb-6 text-gray-900">Full Ecosystem</h3>
                    <p className="text-lg text-gray-600 leading-relaxed font-light">
                        From downloading invoices to filling government forms, writing emails to shopping. Hemlo handles every digital chore across your entire workflow.
                    </p>
                </div>

                {/* Visual */}
                <div className="order-2 relative h-[400px] w-full bg-black rounded-3xl shadow-2xl border border-gray-800 overflow-hidden flex items-center justify-center p-8">
                    {/* Background Grid */}
                    <div className="absolute inset-0 opacity-[0.08]" style={{backgroundImage: 'radial-gradient(#ffffff 1px, transparent 1px)', backgroundSize: '24px 24px'}}></div>

                    {/* Central Core */}
                    <div className="absolute z-20 w-24 h-24 bg-white rounded-2xl flex items-center justify-center shadow-[0_0_60px_rgba(255,255,255,0.15)] animate-pulse-glow-minimal">
                        <span className="font-serif font-bold text-4xl text-black">H</span>
                    </div>

                    {/* Orbiting Tasks Container - Spinning slowly */}
                    <div className="absolute inset-0 animate-spin-slow-smooth">
                        
                        {/* Task 1: Downloading (Top Left) */}
                        <div className="absolute top-1/4 left-1/4 -translate-x-1/2 -translate-y-1/2 w-44 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-3.5 shadow-2xl animate-counter-spin-smooth hover:bg-white/10 transition-all">
                            <div className="flex items-center gap-2 mb-2 text-white/90">
                                <Download size={15} className="animate-bounce-subtle" />
                                <span className="text-[11px] font-mono text-white/70">invoice_2024.pdf</span>
                            </div>
                            <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                                <div className="h-full bg-white/80 w-full animate-progress-bar-smooth origin-left shadow-[0_0_10px_rgba(255,255,255,0.5)]"></div>
                            </div>
                        </div>

                        {/* Task 2: Email (Top Right) */}
                        <div className="absolute top-1/4 right-1/4 translate-x-1/2 -translate-y-1/2 w-44 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-3.5 shadow-2xl animate-counter-spin-smooth hover:bg-white/10 transition-all">
                            <div className="flex items-center gap-2 mb-2 text-white/90">
                                <Mail size={15} className="animate-pulse-subtle" />
                                <span className="text-[11px] font-mono text-white/70">Compose</span>
                            </div>
                            <div className="space-y-1.5">
                                <div className="h-1 bg-white/20 rounded w-full animate-text-typing"></div>
                                <div className="h-1 bg-white/20 rounded w-2/3 animate-text-typing" style={{animationDelay: '0.3s'}}></div>
                                <div className="h-1 bg-white/20 rounded w-3/4 animate-text-typing" style={{animationDelay: '0.6s'}}></div>
                            </div>
                        </div>

                        {/* Task 3: Form Filling (Bottom Right) */}
                        <div className="absolute bottom-1/4 right-1/4 translate-x-1/2 translate-y-1/2 w-44 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-3.5 shadow-2xl animate-counter-spin-smooth hover:bg-white/10 transition-all">
                            <div className="flex items-center gap-2 mb-2 text-white/90">
                                <FileText size={15} />
                                <span className="text-[11px] font-mono text-white/70">Auto-Fill</span>
                            </div>
                            <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                    <div className="w-3.5 h-3.5 rounded-sm bg-white border-2 border-white/30 animate-check-pop-smooth flex items-center justify-center">
                                        <Check size={10} className="text-black opacity-0 animate-check-appear" />
                                    </div>
                                    <div className="h-1 w-20 bg-white/20 rounded"></div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3.5 h-3.5 rounded-sm bg-white border-2 border-white/30 animate-check-pop-smooth" style={{animationDelay: '0.5s'}}>
                                        <Check size={10} className="text-black opacity-0 animate-check-appear" style={{animationDelay: '0.5s'}} />
                                    </div>
                                    <div className="h-1 w-16 bg-white/20 rounded"></div>
                                </div>
                            </div>
                        </div>

                        {/* Task 4: Shopping (Bottom Left) */}
                        <div className="absolute bottom-1/4 left-1/4 -translate-x-1/2 translate-y-1/2 w-44 bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-3.5 shadow-2xl animate-counter-spin-smooth hover:bg-white/10 transition-all">
                            <div className="flex items-center gap-2 mb-2 text-white/90">
                                <ShoppingCart size={15} />
                                <span className="text-[11px] font-mono text-white/70">Checkout</span>
                            </div>
                            <div className="flex items-center justify-between mt-1">
                                <CreditCard size={20} className="text-white/60 animate-card-slide" />
                                <span className="text-[11px] text-white/90 font-bold animate-fade-in-delayed">Paid ✓</span>
                            </div>
                        </div>

                    </div>
                </div>
            </div>

            {/* Feature 3: Magic in Motion (Thinking Robot) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
                {/* Visual */}
                <div className="order-2 lg:order-1 relative h-[400px] w-full bg-black rounded-3xl shadow-2xl border border-gray-800 overflow-hidden flex flex-col">
                    
                    {/* Mock Website in "Dark Mode" */}
                    <div className="absolute inset-4 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
                        {/* Header */}
                        <div className="h-12 bg-gray-800/70 border-b border-gray-700 flex items-center px-4 justify-between">
                            <div className="h-4 w-20 bg-gray-700 rounded animate-pulse-subtle"></div>
                            <div className="flex gap-2">
                                <div className="h-4 w-10 bg-gray-700 rounded"></div>
                                <div className="h-4 w-10 bg-gray-700 rounded"></div>
                            </div>
                        </div>
                        {/* Content */}
                        <div className="p-6 flex flex-col items-center justify-center h-full pb-12 space-y-4">
                            <div className="w-full max-w-xs space-y-2">
                                <div className="text-[10px] text-gray-500 uppercase tracking-widest">Username</div>
                                <div className="h-10 bg-black/50 border border-gray-700 rounded w-full target-element group-1 transition-all duration-300"></div>
                            </div>
                            <div className="w-full max-w-xs space-y-2">
                                <div className="text-[10px] text-gray-500 uppercase tracking-widest">Password</div>
                                <div className="h-10 bg-black/50 border border-gray-700 rounded w-full target-element group-2 transition-all duration-300"></div>
                            </div>
                            <div className="w-full max-w-xs pt-2 flex gap-3">
                                <div className="h-10 flex-1 bg-gray-700 rounded border border-gray-600 target-element group-3"></div>
                                <div className="h-10 flex-1 bg-white rounded target-element group-4 shadow-lg shadow-white/10" id="submit-btn"></div>
                            </div>
                        </div>
                    </div>

                    {/* AI Overlay Layer */}
                    <div className="absolute inset-0 pointer-events-none">
                        
                        {/* Analysis Grid Lines */}
                        <div className="absolute top-0 w-full h-0.5 bg-white/60 shadow-[0_0_20px_rgba(255,255,255,0.6)] animate-scan-down-smooth"></div>

                        {/* Bounding Boxes */}
                        <div className="absolute inset-4 flex flex-col items-center justify-center pb-8 pt-12 space-y-4">
                             {/* Box 1 */}
                             <div className="w-full max-w-xs h-[60px] border border-white/0 animate-box-detect-1-minimal relative">
                                <div className="absolute -top-3 left-0 text-[8px] bg-white text-black px-1.5 py-0.5 font-mono opacity-0 animate-label-fade-1 rounded-sm">INPUT#user</div>
                             </div>
                             {/* Box 2 */}
                             <div className="w-full max-w-xs h-[60px] border border-white/0 animate-box-detect-2-minimal relative">
                                 <div className="absolute -top-3 left-0 text-[8px] bg-white text-black px-1.5 py-0.5 font-mono opacity-0 animate-label-fade-2 rounded-sm">INPUT#pass</div>
                             </div>
                             {/* Box 3 (Buttons) */}
                             <div className="w-full max-w-xs flex gap-3 h-10 mt-6">
                                 <div className="flex-1 border border-white/0 animate-box-detect-3-minimal"></div>
                                 <div className="flex-1 border-2 border-white/0 animate-box-detect-4-minimal relative shadow-[0_0_20px_rgba(255,255,255,0)]">
                                     <div className="absolute -bottom-6 right-0 text-[10px] text-white font-mono font-bold opacity-0 animate-label-success-minimal">CONFIDENCE: 99%</div>
                                 </div>
                             </div>
                        </div>

                        {/* The Brain / Logic Node */}
                        <div className="absolute top-8 right-8 bg-black/90 backdrop-blur-sm border border-white/20 p-3 rounded-lg flex items-center gap-3 animate-fade-in-smooth text-white shadow-[0_0_40px_rgba(255,255,255,0.15)]">
                             <Cpu className="animate-pulse-subtle" size={18} />
                             <div className="text-[10px] font-mono leading-tight">
                                 <div className="font-bold">DOM_ANALYSIS</div>
                                 <div className="text-gray-400 animate-text-cycle-minimal">Scanning...</div>
                             </div>
                        </div>

                        {/* Cursor Action */}
                        <div className="absolute z-50 animate-cursor-logic-smooth opacity-0">
                            <MousePointer size={24} className="fill-white text-black drop-shadow-[0_0_15px_rgba(255,255,255,0.8)]" />
                        </div>

                    </div>
                </div>

                {/* Text */}
                <div className="order-1 lg:order-2 lg:pl-8">
                    <h3 className="font-serif text-3xl md:text-4xl font-bold mb-6 text-gray-900">Magic in Motion</h3>
                    <p className="text-lg text-gray-600 leading-relaxed font-light">
                        It doesn't just guess. It analyzes the webpage DOM, identifies the correct elements ("Submit", "Buy Now"), thinks about the workflow, and clicks exactly like a human.
                    </p>
                </div>
            </div>

        </div>

        {/* Animation Styles */}
        <style>{`
            /* Global Utilities - Ultra Smooth Rotation with Ease */
            .animate-spin-slow-smooth { animation: spin 35s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite; }
            .animate-counter-spin-smooth { animation: spin-reverse 35s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite; }
            @keyframes spin { 
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); } 
            }
            @keyframes spin-reverse { 
                0% { transform: rotate(0deg); }
                100% { transform: rotate(-360deg); } 
            }

            /* --- Section 1: Browser Automation Sequence (8s Loop) --- */
            
            /* Prompt typing */
            .animate-typing-text {
                width: 0;
                animation: typing 2s steps(20, end) forwards infinite;
                animation-iteration-count: 1; /* Triggered by parent loop concept ideally, using delays here for CSS only flow */
            }
            @keyframes typing { 
                0%, 10% { width: 0; }
                50%, 90% { width: 100%; }
                100% { width: 100%; } 
            }

            /* Container Fade/Slide Logic for Seq 1 */
            .animate-prompt-sequence {
                animation: prompt-fade 8s infinite;
            }
            @keyframes prompt-fade {
                0%, 20% { opacity: 1; transform: translate(-50%, 0); }
                25%, 100% { opacity: 0; transform: translate(-50%, -20px); }
            }

            .animate-browser-sequence {
                animation: browser-slide 8s infinite;
            }
            @keyframes browser-slide {
                0%, 20% { opacity: 0; transform: translateY(40px); }
                25%, 90% { opacity: 1; transform: translateY(0); }
                100% { opacity: 0; transform: translateY(-10px); }
            }

            .animate-fill-from { animation: text-appear 8s infinite; animation-delay: 2.5s; }
            .animate-fill-to { animation: text-appear 8s infinite; animation-delay: 3s; }
            
            @keyframes text-appear {
                0% { opacity: 0; }
                5%, 80% { opacity: 1; }
                100% { opacity: 0; }
            }

            /* Cursor Movement in Browser */
            .animate-cursor-flow {
                animation: cursor-path 8s infinite;
            }
            @keyframes cursor-path {
                0%, 30% { opacity: 0; top: 110%; left: 50%; }
                35% { opacity: 1; top: 80%; left: 80%; } /* Move to button */
                40% { transform: scale(0.9); } /* Click */
                45% { transform: scale(1); opacity: 1; }
                50%, 100% { opacity: 0; top: 80%; left: 80%; } /* Disappear after click */
            }

            .animate-show-results {
                animation: results-pop 8s infinite;
                animation-delay: 4.5s;
            }
            @keyframes results-pop {
                0% { opacity: 0; transform: translateY(10px); }
                10%, 80% { opacity: 1; transform: translateY(0); }
                100% { opacity: 0; }
            }


            /* --- Section 2: Ecosystem Loop - Ultra Smooth Detailed Animations --- */
            .animate-pulse-glow-minimal { animation: pulse-glow-minimal 5s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite; }
            @keyframes pulse-glow-minimal {
                0% { box-shadow: 0 0 40px rgba(255,255,255,0.08), 0 0 80px rgba(255,255,255,0.04); transform: scale(1); }
                15% { box-shadow: 0 0 45px rgba(255,255,255,0.12), 0 0 85px rgba(255,255,255,0.06); transform: scale(1.008); }
                50% { box-shadow: 0 0 65px rgba(255,255,255,0.22), 0 0 110px rgba(255,255,255,0.12); transform: scale(1.025); }
                85% { box-shadow: 0 0 45px rgba(255,255,255,0.12), 0 0 85px rgba(255,255,255,0.06); transform: scale(1.008); }
                100% { box-shadow: 0 0 40px rgba(255,255,255,0.08), 0 0 80px rgba(255,255,255,0.04); transform: scale(1); }
            }

            .animate-progress-bar-smooth { animation: progress-smooth 4s cubic-bezier(0.25, 0.46, 0.45, 0.94) infinite; }
            @keyframes progress-smooth {
                0% { width: 0; opacity: 0.5; transform: scaleX(0); transform-origin: left; }
                5% { opacity: 0.7; }
                15% { width: 20%; transform: scaleX(1); }
                40% { width: 60%; opacity: 1; }
                70% { width: 90%; opacity: 0.95; }
                90%, 100% { width: 100%; opacity: 0.85; }
            }

            .animate-bounce-subtle { animation: bounce-subtle 2.5s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite; }
            @keyframes bounce-subtle {
                0%, 100% { transform: translateY(0); }
                25% { transform: translateY(-2px); }
                50% { transform: translateY(-4px); }
                75% { transform: translateY(-2px); }
            }

            .animate-pulse-subtle { animation: pulse-subtle 3.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
            @keyframes pulse-subtle {
                0%, 100% { opacity: 0.85; transform: scale(1); }
                50% { opacity: 1; transform: scale(1.05); }
            }

            .animate-text-typing { animation: text-typing 5s cubic-bezier(0.4, 0, 0.2, 1) infinite; }
            @keyframes text-typing {
                0% { width: 0; opacity: 0; }
                8% { opacity: 0.4; }
                12% { width: 15%; opacity: 0.7; }
                20% { width: 35%; opacity: 0.9; }
                30% { width: 65%; opacity: 1; }
                45%, 75% { width: 100%; opacity: 1; }
                85% { width: 100%; opacity: 0.6; }
                100% { width: 100%; opacity: 0.2; }
            }

            .animate-check-pop-smooth { animation: check-scale-smooth 3.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite; }
            @keyframes check-scale-smooth {
                0%, 68% { transform: scale(0) rotate(-45deg); opacity: 0; }
                72% { transform: scale(0.5) rotate(-22deg); opacity: 0.5; }
                76% { transform: scale(1.2) rotate(5deg); opacity: 1; }
                80% { transform: scale(0.95) rotate(-2deg); opacity: 1; }
                84% { transform: scale(1.05) rotate(1deg); opacity: 1; }
                88%, 100% { transform: scale(1) rotate(0deg); opacity: 1; }
            }

            .animate-check-appear { animation: check-appear 3.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) infinite; }
            @keyframes check-appear {
                0%, 72% { opacity: 0; transform: scale(0) rotate(-90deg); }
                76% { opacity: 0.6; transform: scale(0.8) rotate(-10deg); }
                80% { opacity: 1; transform: scale(1.3) rotate(5deg); }
                84% { opacity: 1; transform: scale(0.9) rotate(-2deg); }
                88%, 100% { opacity: 1; transform: scale(1) rotate(0deg); }
            }

            .animate-card-slide { animation: card-slide 4.5s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite; }
            @keyframes card-slide {
                0%, 25% { transform: translateX(0) rotate(0deg); opacity: 0.5; }
                30% { transform: translateX(2px) rotate(1deg); opacity: 0.7; }
                35% { transform: translateX(5px) rotate(2deg); opacity: 0.85; }
                45%, 65% { transform: translateX(10px) rotate(3deg); opacity: 1; }
                70% { transform: translateX(5px) rotate(1deg); opacity: 0.85; }
                80%, 100% { transform: translateX(0) rotate(0deg); opacity: 0.5; }
            }

            .animate-fade-in-delayed { animation: fade-in-delayed 4.5s cubic-bezier(0.4, 0, 0.2, 1) infinite; }
            @keyframes fade-in-delayed {
                0%, 38% { opacity: 0; transform: scale(0.7) translateY(5px); }
                42% { opacity: 0.3; transform: scale(0.85) translateY(2px); }
                48% { opacity: 0.7; transform: scale(0.95) translateY(0); }
                52%, 88% { opacity: 1; transform: scale(1) translateY(0); }
                92% { opacity: 0.6; transform: scale(0.98) translateY(1px); }
                100% { opacity: 0; transform: scale(0.9) translateY(3px); }
            }


            /* --- Section 3: Logic/DOM Loop - Ultra Detailed (8s) --- */
            
            .animate-scan-down-smooth {
                animation: scan-line-smooth 8s cubic-bezier(0.25, 0.46, 0.45, 0.94) infinite;
            }
            @keyframes scan-line-smooth {
                0% { top: 0%; opacity: 0; height: 0.5px; box-shadow: 0 0 10px rgba(255,255,255,0.3); }
                3% { opacity: 0.3; height: 1px; box-shadow: 0 0 15px rgba(255,255,255,0.5); }
                8% { opacity: 0.9; height: 1px; box-shadow: 0 0 25px rgba(255,255,255,0.8); }
                12% { top: 15%; opacity: 1; }
                25% { top: 40%; opacity: 1; }
                40% { top: 75%; opacity: 1; }
                48% { top: 100%; opacity: 0.8; height: 0.5px; box-shadow: 0 0 20px rgba(255,255,255,0.6); }
                55%, 100% { top: 100%; opacity: 0; height: 0.5px; box-shadow: none; }
            }

            .animate-box-detect-1-minimal { animation: box-border-minimal 8s cubic-bezier(0.4, 0, 0.2, 1) infinite; animation-delay: 1.1s; }
            .animate-box-detect-2-minimal { animation: box-border-minimal 8s cubic-bezier(0.4, 0, 0.2, 1) infinite; animation-delay: 1.8s; }
            .animate-box-detect-3-minimal { animation: box-border-minimal 8s cubic-bezier(0.4, 0, 0.2, 1) infinite; animation-delay: 2.5s; }
            
            @keyframes box-border-minimal {
                0% { border-color: rgba(255,255,255,0); background: transparent; transform: scale(1); }
                3% { border-color: rgba(255,255,255,0.2); background: rgba(255,255,255,0.02); transform: scale(1.005); }
                6% { border-color: rgba(255,255,255,0.5); background: rgba(255,255,255,0.06); transform: scale(1.01); }
                12% { border-color: rgba(255,255,255,0.7); background: rgba(255,255,255,0.1); transform: scale(1.015); }
                25% { border-color: rgba(255,255,255,0.65); background: rgba(255,255,255,0.09); transform: scale(1.01); }
                30% { border-color: rgba(255,255,255,0.4); background: rgba(255,255,255,0.04); transform: scale(1.005); }
                35%, 100% { border-color: rgba(255,255,255,0); background: transparent; transform: scale(1); }
            }
            
            .animate-label-fade-1 { animation: fade-quick-smooth 8s cubic-bezier(0.4, 0, 0.2, 1) infinite; animation-delay: 1.1s; }
            .animate-label-fade-2 { animation: fade-quick-smooth 8s cubic-bezier(0.4, 0, 0.2, 1) infinite; animation-delay: 1.8s; }
            
            @keyframes fade-quick-smooth {
                0% { opacity: 0; transform: translateY(4px) scale(0.9); }
                3% { opacity: 0.3; transform: translateY(2px) scale(0.95); }
                6% { opacity: 0.7; transform: translateY(0) scale(1); }
                25% { opacity: 1; transform: translateY(0) scale(1); }
                30% { opacity: 0.6; transform: translateY(-1px) scale(0.98); }
                35%, 100% { opacity: 0; transform: translateY(-3px) scale(0.95); }
            }

            /* Correct Target Animation - Progressive White Glow */
            .animate-box-detect-4-minimal { animation: box-success-minimal 8s cubic-bezier(0.4, 0, 0.2, 1) infinite; animation-delay: 3.2s; }
            @keyframes box-success-minimal {
                0% { border-color: transparent; box-shadow: none; transform: scale(1); }
                3% { border-color: rgba(255,255,255,0.3); box-shadow: 0 0 10px rgba(255,255,255,0.15), inset 0 0 8px rgba(255,255,255,0.08); transform: scale(1.008); }
                6% { border-color: rgba(255,255,255,0.6); box-shadow: 0 0 20px rgba(255,255,255,0.3), inset 0 0 12px rgba(255,255,255,0.15); transform: scale(1.015); }
                10% { border-color: rgba(255,255,255,0.85); box-shadow: 0 0 30px rgba(255,255,255,0.45), inset 0 0 18px rgba(255,255,255,0.22); transform: scale(1.022); }
                20% { border-color: rgba(255,255,255,0.95); box-shadow: 0 0 35px rgba(255,255,255,0.55), 0 0 60px rgba(255,255,255,0.25), inset 0 0 22px rgba(255,255,255,0.28); transform: scale(1.025); }
                60% { border-color: rgba(255,255,255,0.95); box-shadow: 0 0 35px rgba(255,255,255,0.55), 0 0 60px rgba(255,255,255,0.25), inset 0 0 22px rgba(255,255,255,0.28); transform: scale(1.025); }
                68% { border-color: rgba(255,255,255,0.7); box-shadow: 0 0 25px rgba(255,255,255,0.35), inset 0 0 15px rgba(255,255,255,0.18); transform: scale(1.015); }
                75% { border-color: rgba(255,255,255,0.3); box-shadow: 0 0 12px rgba(255,255,255,0.2); transform: scale(1.005); }
                82%, 100% { border-color: transparent; box-shadow: none; transform: scale(1); }
            }
            
            .animate-label-success-minimal { animation: fade-stay-smooth 8s cubic-bezier(0.4, 0, 0.2, 1) infinite; animation-delay: 3.2s; }
            @keyframes fade-stay-smooth {
                0% { opacity: 0; transform: scale(0.85) translateX(5px); }
                3% { opacity: 0.3; transform: scale(0.9) translateX(3px); }
                8% { opacity: 0.7; transform: scale(0.97) translateX(1px); }
                12% { opacity: 1; transform: scale(1.05) translateX(0); }
                16%, 65% { opacity: 1; transform: scale(1) translateX(0); }
                70% { opacity: 0.8; transform: scale(0.98) translateX(-1px); }
                75% { opacity: 0.5; transform: scale(0.95) translateX(-2px); }
                82%, 100% { opacity: 0; transform: scale(0.9) translateX(-3px); }
            }

            /* Logic Text Cycle - Smooth Thinking */
            .animate-text-cycle-minimal { animation: text-change-minimal 8s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
            @keyframes text-change-minimal {
                0%, 18% { opacity: 0.65; transform: translateX(0); }
                22% { opacity: 0.5; transform: translateX(-2px); }
                26% { opacity: 0.3; transform: translateX(-3px); }
                30% { opacity: 0; transform: translateX(-4px); }
                34% { opacity: 0.2; transform: translateX(3px); }
                38%, 68% { opacity: 1; transform: translateX(0); }
                72% { opacity: 0.7; transform: translateX(-1px); }
                78%, 100% { opacity: 0; transform: translateX(-2px); }
            }

            .animate-fade-in-smooth { animation: fade-in-brain 8s cubic-bezier(0.34, 1.56, 0.64, 1) infinite; }
            @keyframes fade-in-brain {
                0% { opacity: 0; transform: scale(0.92) translateY(3px); }
                4% { opacity: 0.3; transform: scale(0.96) translateY(1px); }
                8% { opacity: 0.7; transform: scale(1.02) translateY(0); }
                12%, 82% { opacity: 1; transform: scale(1) translateY(0); }
                88% { opacity: 0.6; transform: scale(0.98) translateY(1px); }
                100% { opacity: 0; transform: scale(0.95) translateY(2px); }
            }

            /* Cursor Logic Move - Natural Bezier Path */
            .animate-cursor-logic-smooth { animation: cursor-logic-move-smooth 8s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite; }
            @keyframes cursor-logic-move-smooth {
                0%, 26% { opacity: 0; top: 8%; right: 8%; transform: scale(0.8) rotate(-5deg); }
                30% { opacity: 0.5; top: 9%; right: 9%; transform: scale(0.95) rotate(-2deg); }
                34% { opacity: 1; top: 10%; right: 10%; transform: scale(1) rotate(0deg); }
                38% { top: 25%; right: 12%; transform: scale(1) rotate(8deg); }
                42% { top: 45%; right: 15%; transform: scale(1) rotate(15deg); }
                46% { top: 65%; right: 18%; transform: scale(1) rotate(20deg); }
                50% { top: 75%; right: 20%; transform: scale(1.05) rotate(22deg); }
                54% { top: 75%; right: 20%; transform: scale(0.82) rotate(22deg); }
                57% { top: 75%; right: 20%; transform: scale(1.08) rotate(22deg); }
                60% { top: 75%; right: 20%; transform: scale(1) rotate(22deg); }
                72%, 100% { opacity: 0; top: 75%; right: 20%; transform: scale(0.9) rotate(22deg); }
            }

        `}</style>
    </section>
  );
};

export default Vision;