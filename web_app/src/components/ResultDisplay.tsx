import { useEffect, useRef } from 'react';
import { Volume2, Clock } from 'lucide-react';
import { RecognitionResult } from '../services/recognitionService';

interface ResultDisplayProps {
  latestResult: RecognitionResult | null;
  recentResults: RecognitionResult[];
}

const ResultDisplay = ({ latestResult, recentResults }: ResultDisplayProps) => {
  const latestResultRef = useRef<HTMLDivElement>(null);
  
  // Scroll to the latest result when it updates
  useEffect(() => {
    if (latestResult && latestResultRef.current) {
      latestResultRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [latestResult]);
  
  // Format timestamp
  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };
  
  return (
    <div className="glass-card h-full flex flex-col">
      <div className="p-4 border-b border-border">
        <h3 className="text-lg font-medium">Recognition Results</h3>
      </div>
      
      {/* Latest result */}
      <div className="p-4 border-b border-border bg-primary/5">
        <div className="text-sm font-medium text-muted-foreground mb-2">Latest Recognition</div>
        
        {latestResult ? (
          <div className="animate-scale-in" ref={latestResultRef}>
            <div className="text-3xl font-bold mb-4">
              {latestResult.gesture}
            </div>
            
            <div className="flex justify-between text-sm text-muted-foreground">
              <div className="flex items-center">
                <Clock className="h-3.5 w-3.5 mr-1.5" />
                <span>{formatTime(latestResult.timestamp)}</span>
              </div>
            </div>
            
            <button 
              className="mt-4 w-full btn-secondary py-1 flex items-center justify-center"
              onClick={() => {
                if (latestResult) {
                  const utterance = new SpeechSynthesisUtterance(latestResult.gesture);
                  utterance.rate = 1.2;
                  window.speechSynthesis.cancel(); // Prevents queuing delays
                  window.speechSynthesis.speak(utterance);
                }
              }}
            >
              <Volume2 className="h-4 w-4 mr-2" />
              Speak Again
            </button>
          </div>
        ) : (
          <div className="text-muted-foreground text-center py-4">
            No gestures recognized yet. Start the recognition to begin.
          </div>
        )}
      </div>
      
      {/* Recent history - Keep same height as camera */}
      <div className="flex-grow overflow-y-auto">
        <div className="p-4">
          <div className="text-sm font-medium text-muted-foreground mb-2">Recent History</div>
          
          {recentResults.length > 0 ? (
            <ul className="space-y-2">
              {recentResults.map((result, index) => (
                <li 
                  key={`${result.timestamp}-${index}`}
                  className="glass-card p-2 animate-fade-in"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-medium">{result.gesture}</div>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {formatTime(result.timestamp)}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-muted-foreground text-center py-3">
              No recent recognition history.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultDisplay;
