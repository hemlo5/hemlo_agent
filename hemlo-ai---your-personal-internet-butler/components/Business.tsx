import React from 'react';
import { DollarSign, Trophy, Clock, Check, X } from 'lucide-react';

const Business: React.FC = () => {
  return (
    <section className="w-full py-24 bg-white relative">
      <div className="max-w-7xl mx-auto px-6">
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
          
          {/* Left Content */}
          <div className="space-y-12">
            <h2 className="font-serif text-4xl md:text-[40px] leading-tight font-bold text-gray-900">
              The Opportunity: <br/>
              <span className="text-yellow-600">$47B Market</span>, Hemlo Leads
            </h2>

            <div className="space-y-8">
              {/* Item 1 */}
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-yellow-50 flex items-center justify-center text-yellow-600">
                  <DollarSign size={20} />
                </div>
                <div>
                  <h4 className="text-lg font-bold text-gray-900 mb-2">Market Size</h4>
                  <p className="text-gray-600 leading-relaxed">
                    AI agents market hits $47B by 2030 – 2B internet users need this. Hemlo captures the 'daily digital chores' slice (refunds, bookings, social posts).
                  </p>
                </div>
              </div>

              {/* Item 2 */}
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-yellow-50 flex items-center justify-center text-yellow-600">
                  <Trophy size={20} />
                </div>
                <div>
                  <h4 className="text-lg font-bold text-gray-900 mb-2">Competitors</h4>
                  <p className="text-gray-600 leading-relaxed">
                    Tools like Comet (slow, vision-only) and Bhindi (app APIs only) exist, but Hemlo's DOM magic works on ANY website without integrations – faster, cheaper, private.
                  </p>
                </div>
              </div>

              {/* Item 3 */}
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-yellow-50 flex items-center justify-center text-yellow-600">
                  <Clock size={20} />
                </div>
                <div>
                  <h4 className="text-lg font-bold text-gray-900 mb-2">First Mover Advantage</h4>
                  <p className="text-gray-600 leading-relaxed">
                    We're teens building the real thing: local-first, no data leaks, adapts instantly. Launching US/EU first ($19/mo), then India (₹99/mo). Early beta users get lifetime free.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Visuals: Chart & Table */}
          <div className="bg-gray-50 rounded-3xl p-8 border border-gray-100 shadow-xl shadow-gray-200/50">
            
            {/* Comparison Table */}
            <div className="mb-10">
              <h3 className="font-serif text-xl font-bold text-gray-900 mb-6">Why Hemlo Wins</h3>
              <div className="overflow-hidden rounded-xl border border-gray-200 bg-white">
                <table className="w-full text-left text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="py-3 px-4 font-semibold text-gray-500">Feature</th>
                      <th className="py-3 px-4 font-semibold text-gray-500 text-center">Competitors</th>
                      <th className="py-3 px-4 font-bold text-yellow-700 bg-yellow-50 text-center">Hemlo</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    <tr>
                      <td className="py-3 px-4 text-gray-800 font-medium">Speed</td>
                      <td className="py-3 px-4 text-gray-500 text-center">Slow (Vision)</td>
                      <td className="py-3 px-4 text-gray-900 text-center font-bold bg-yellow-50/50">10x (DOM)</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 text-gray-800 font-medium">Privacy</td>
                      <td className="py-3 px-4 text-gray-500 text-center">Cloud</td>
                      <td className="py-3 px-4 text-gray-900 text-center font-bold bg-yellow-50/50">Local-First</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 text-gray-800 font-medium">Cost</td>
                      <td className="py-3 px-4 text-gray-500 text-center">High ($30+)</td>
                      <td className="py-3 px-4 text-gray-900 text-center font-bold bg-yellow-50/50">Low ($19)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Market Growth Chart (Simulated) */}
            <div>
              <div className="flex justify-between items-end mb-2">
                <h3 className="font-serif text-lg font-bold text-gray-900">Market Growth</h3>
                <span className="text-xs font-semibold text-yellow-600 bg-yellow-50 px-2 py-1 rounded-full">+47B Opportunity</span>
              </div>
              
              <div className="h-40 flex items-end gap-2 pt-4 border-b border-gray-200 relative">
                 {/* Grid lines */}
                 <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
                    <div className="w-full h-px bg-gray-100 border-t border-dashed border-gray-200"></div>
                    <div className="w-full h-px bg-gray-100 border-t border-dashed border-gray-200"></div>
                    <div className="w-full h-px bg-gray-100 border-t border-dashed border-gray-200"></div>
                 </div>

                 {/* Bars */}
                 <div className="w-full bg-gray-200 rounded-t-sm h-[20%] relative group">
                    <div className="opacity-0 group-hover:opacity-100 absolute -top-8 left-1/2 -translate-x-1/2 bg-gray-800 text-white text-xs px-2 py-1 rounded transition-opacity">2023</div>
                 </div>
                 <div className="w-full bg-gray-300 rounded-t-sm h-[35%] relative group">
                    <div className="opacity-0 group-hover:opacity-100 absolute -top-8 left-1/2 -translate-x-1/2 bg-gray-800 text-white text-xs px-2 py-1 rounded transition-opacity">2025</div>
                 </div>
                 <div className="w-full bg-gray-400 rounded-t-sm h-[60%] relative group">
                    <div className="opacity-0 group-hover:opacity-100 absolute -top-8 left-1/2 -translate-x-1/2 bg-gray-800 text-white text-xs px-2 py-1 rounded transition-opacity">2028</div>
                 </div>
                 <div className="w-full bg-yellow-500 rounded-t-sm h-[90%] relative group shadow-lg shadow-yellow-500/20">
                    <div className="opacity-0 group-hover:opacity-100 absolute -top-8 left-1/2 -translate-x-1/2 bg-gray-800 text-white text-xs px-2 py-1 rounded transition-opacity">2030</div>
                 </div>
              </div>
            </div>

          </div>

        </div>
      </div>
    </section>
  );
};

export default Business;